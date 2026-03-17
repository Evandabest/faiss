// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIVFFlat.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/DirectMap.h>

namespace faiss {
namespace gpu_metal {

namespace {
inline uint32_t checkedToU32(size_t v, const char* what) {
    if (v > (size_t)std::numeric_limits<uint32_t>::max()) {
        FAISS_THROW_MSG(what);
    }
    return static_cast<uint32_t>(v);
}
} // namespace

MetalIVFFlatImpl::MetalIVFFlatImpl(
        std::shared_ptr<MetalResources> resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        faiss::gpu::IndicesOptions indicesOptions,
        bool interleavedLayout)
        : resources_(std::move(resources)),
          dim_(dim),
          nlist_(nlist),
          metric_type_(metric),
          metric_arg_(metricArg),
          indicesOptions_(indicesOptions),
          interleavedLayout_(interleavedLayout),
          listLength_(nlist_, 0),
          listOffset_(nlist_, 0),
          listCapacity_(nlist_, 0),
          totalVecs_(0),
          totalCapacityVecs_(0),
          codesBuffer_(nil),
          idsBuffer_(nil),
          listOffsetBuf_(nil),
          listLengthBuf_(nil),
          interleavedCodesBuf_(nil),
          interleavedCodesOffsetBuf_(nil) {
    FAISS_THROW_IF_NOT(dim_ > 0);
    FAISS_THROW_IF_NOT(nlist_ >= 0);
}

MetalIVFFlatImpl::~MetalIVFFlatImpl() {
    reset();
}

void MetalIVFFlatImpl::reset() {
    hostCodes_.clear();
    hostIds_.clear();
    totalVecs_ = 0;
    totalCapacityVecs_ = 0;

    std::fill(listLength_.begin(), listLength_.end(), 0);
    std::fill(listOffset_.begin(), listOffset_.end(), 0);
    std::fill(listCapacity_.begin(), listCapacity_.end(), 0);

    if (codesBuffer_ != nil) {
        resources_->deallocBuffer(codesBuffer_, MetalAllocType::IVFLists);
        codesBuffer_ = nil;
    }
    if (idsBuffer_ != nil) {
        resources_->deallocBuffer(idsBuffer_, MetalAllocType::IVFLists);
        idsBuffer_ = nil;
    }
    if (listOffsetBuf_ != nil) {
        resources_->deallocBuffer(listOffsetBuf_, MetalAllocType::IVFLists);
        listOffsetBuf_ = nil;
    }
    if (listLengthBuf_ != nil) {
        resources_->deallocBuffer(listLengthBuf_, MetalAllocType::IVFLists);
        listLengthBuf_ = nil;
    }
    if (interleavedCodesBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
        interleavedCodesBuf_ = nil;
    }
    if (interleavedCodesOffsetBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
        interleavedCodesOffsetBuf_ = nil;
    }
}

void MetalIVFFlatImpl::reserveMemory(idx_t totalVecs) {
    if (totalVecs <= 0) {
        return;
    }
    size_t t = (size_t)totalVecs;
    if (t <= totalVecs_) {
        return;
    }
    // Distribute reserve evenly across lists and over-allocate by one on the first
    // rem lists to preserve total target.
    size_t base = (nlist_ > 0) ? (t / (size_t)nlist_) : 0;
    size_t rem = (nlist_ > 0) ? (t % (size_t)nlist_) : 0;

    std::vector<size_t> reservePerList((size_t)nlist_, 0);
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t target = base + (l < rem ? 1 : 0);
        reservePerList[l] =
                target > listLength_[l] ? (target - listLength_[l]) : 0;
    }
    ensureCapacityForAppend_(reservePerList);

    // Materialize GPU allocations up-front so future appends can avoid
    // reallocating until reserved capacity is exceeded.
    std::vector<size_t> oldLength((size_t)nlist_, 0);
    std::vector<size_t> addPerList((size_t)nlist_, 0);
    uploadToGpu_(oldLength, addPerList, true);
}

void MetalIVFFlatImpl::appendVectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        const idx_t* xids) {
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(list_nos != nullptr);

    // Count how many vectors go to each list.
    std::vector<size_t> addPerList(nlist_, 0);
    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_) {
            continue;
        }
        addPerList[(size_t)list]++;
    }

    // Early-out: nothing to append.
    size_t batchNew = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        batchNew += addPerList[l];
    }
    if (batchNew == 0) {
        return;
    }

    const std::vector<size_t> oldLength = listLength_;
    bool forceFullUpload = ensureCapacityForAppend_(addPerList);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_) {
            continue;
        }
        size_t l = (size_t)list;
        size_t dstIndex = listOffset_[l] + listLength_[l];
        size_t listOffset = listLength_[l];
        listLength_[l]++;

        // Copy vector
        const float* xi = x + (size_t)i * (size_t)dim_;
        std::memcpy(
                hostCodes_.data() + dstIndex * (size_t)dim_,
                xi,
                (size_t)dim_ * sizeof(float));

        // Copy id
        idx_t id = -1;
        if (indicesOptions_ == faiss::gpu::INDICES_CPU ||
            indicesOptions_ == faiss::gpu::INDICES_IVF) {
            id = (idx_t)faiss::lo_build((uint64_t)l, (uint64_t)listOffset);
        } else {
            id = xids ? xids[i] : (idx_t)(totalVecs_ + (size_t)(i));
            if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                FAISS_THROW_IF_NOT_MSG(
                        id >= (idx_t)std::numeric_limits<int32_t>::min() &&
                                id <= (idx_t)std::numeric_limits<int32_t>::max(),
                        "MetalIVFFlatImpl: id out of int32 range");
                id = (idx_t)(int32_t)id;
            }
        }
        hostIds_[dstIndex] = id;
    }

    totalVecs_ += batchNew;
    uploadToGpu_(oldLength, addPerList, forceFullUpload);
}

bool MetalIVFFlatImpl::ensureCapacityForAppend_(
        const std::vector<size_t>& addPerList) {
    std::vector<size_t> newCapacity = listCapacity_;
    bool needsRelayout = false;

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t need = listLength_[l] + addPerList[l];
        if (need <= newCapacity[l]) {
            continue;
        }

        size_t cap = std::max<size_t>(1, newCapacity[l]);
        while (cap < need) {
            cap *= 2;
        }
        newCapacity[l] = cap;
        needsRelayout = true;
    }

    if (!needsRelayout) {
        return false;
    }

    std::vector<size_t> newOffset((size_t)nlist_, 0);
    size_t prefix = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        newOffset[l] = prefix;
        prefix += newCapacity[l];
    }
    size_t newTotalCapacity = prefix;

    std::vector<float> newCodes(newTotalCapacity * (size_t)dim_, 0.0f);
    std::vector<idx_t> newIds(newTotalCapacity, (idx_t)-1);

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t len = listLength_[l];
        if (len == 0) {
            continue;
        }
        size_t oldOff = listOffset_[l];
        size_t newOff = newOffset[l];
        std::memcpy(
                newCodes.data() + newOff * (size_t)dim_,
                hostCodes_.data() + oldOff * (size_t)dim_,
                len * (size_t)dim_ * sizeof(float));
        std::memcpy(
                newIds.data() + newOff,
                hostIds_.data() + oldOff,
                len * sizeof(idx_t));
    }

    hostCodes_.swap(newCodes);
    hostIds_.swap(newIds);
    listOffset_.swap(newOffset);
    listCapacity_.swap(newCapacity);
    totalCapacityVecs_ = newTotalCapacity;

    return true;
}

void MetalIVFFlatImpl::uploadToGpu_(
        const std::vector<size_t>& oldLength,
        const std::vector<size_t>& addPerList,
        bool forceFullUpload) {
    if (!resources_ || !resources_->isAvailable()) {
        return;
    }

    size_t codesBytes = totalCapacityVecs_ * (size_t)dim_ * sizeof(float);
    size_t idsBytes = totalCapacityVecs_ * sizeof(idx_t);
    size_t metaBytes  = (size_t)nlist_ * sizeof(uint32_t);
    FAISS_THROW_IF_NOT_MSG(
            (size_t)nlist_ <= (size_t)std::numeric_limits<uint32_t>::max(),
            "MetalIVFFlatImpl: nlist exceeds uint32 metadata range");

    // Always update metadata buffers — they reflect current list layout.
    if (metaBytes > 0) {
        if (listOffsetBuf_ == nil || [listOffsetBuf_ length] < metaBytes) {
            if (listOffsetBuf_ != nil) {
                resources_->deallocBuffer(listOffsetBuf_, MetalAllocType::IVFLists);
            }
            listOffsetBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        }
        if (listLengthBuf_ == nil || [listLengthBuf_ length] < metaBytes) {
            if (listLengthBuf_ != nil) {
                resources_->deallocBuffer(listLengthBuf_, MetalAllocType::IVFLists);
            }
            listLengthBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        }
        FAISS_THROW_IF_NOT_MSG(
                listOffsetBuf_ && listLengthBuf_,
                "MetalIVFFlatImpl: failed to allocate metadata GPU buffers");

        auto* offPtr = reinterpret_cast<uint32_t*>([listOffsetBuf_ contents]);
        auto* lenPtr = reinterpret_cast<uint32_t*>([listLengthBuf_ contents]);
        for (size_t i = 0; i < (size_t)nlist_; ++i) {
            offPtr[i] = checkedToU32(
                    listOffset_[i],
                    "MetalIVFFlatImpl: list offset exceeds uint32 range");
            lenPtr[i] = checkedToU32(
                    listLength_[i],
                    "MetalIVFFlatImpl: list length exceeds uint32 range");
        }
    }

    if (codesBytes == 0 || idsBytes == 0) {
        return;
    }

    if (idsBuffer_ == nil || [idsBuffer_ length] < idsBytes) {
        if (idsBuffer_ != nil) {
            resources_->deallocBuffer(idsBuffer_, MetalAllocType::IVFLists);
        }
        idsBuffer_ = resources_->allocBuffer(idsBytes, MetalAllocType::IVFLists);
        forceFullUpload = true;
    }

    FAISS_THROW_IF_NOT_MSG(
            idsBuffer_,
            "MetalIVFFlatImpl: failed to allocate IVF ids buffer");

    if (!interleavedLayout_) {
        if (codesBuffer_ == nil || [codesBuffer_ length] < codesBytes) {
            if (codesBuffer_ != nil) {
                resources_->deallocBuffer(codesBuffer_, MetalAllocType::IVFLists);
            }
            codesBuffer_ = resources_->allocBuffer(codesBytes, MetalAllocType::IVFLists);
            forceFullUpload = true;
        }

        FAISS_THROW_IF_NOT_MSG(
                codesBuffer_,
                "MetalIVFFlatImpl: failed to allocate IVF codes buffer");
    }

    if (forceFullUpload || interleavedLayout_) {
        std::memcpy([idsBuffer_ contents], hostIds_.data(), idsBytes);
        if (!interleavedLayout_) {
            std::memcpy([codesBuffer_ contents], hostCodes_.data(), codesBytes);
        }
    } else {
        for (size_t l = 0; l < (size_t)nlist_; ++l) {
            size_t add = addPerList[l];
            if (add == 0) {
                continue;
            }
            size_t start = listOffset_[l] + oldLength[l];

            std::memcpy(
                    reinterpret_cast<idx_t*>([idsBuffer_ contents]) + start,
                    hostIds_.data() + start,
                    add * sizeof(idx_t));
            std::memcpy(
                    reinterpret_cast<float*>([codesBuffer_ contents]) +
                            start * (size_t)dim_,
                    hostCodes_.data() + start * (size_t)dim_,
                    add * (size_t)dim_ * sizeof(float));
        }
    }

    bool haveInterleaved = false;
    if (interleavedLayout_) {
        // Build interleaved codes buffer: blocks of 32 vectors with dims interleaved.
        // Layout per block: [v0d0 v1d0 ... v31d0] [v0d1 v1d1 ... v31d1] ...
        constexpr int G = kInterleavedGroupSize; // 32
        std::vector<uint32_t> ilOffsets((size_t)nlist_);
        size_t totalIlFloats = 0;
        for (size_t l = 0; l < (size_t)nlist_; ++l) {
            ilOffsets[l] = checkedToU32(
                    totalIlFloats,
                    "MetalIVFFlatImpl: interleaved offset exceeds uint32 range");
            size_t len = listLength_[l];
            size_t numBlocks = (len + G - 1) / G;
            totalIlFloats += numBlocks * G * (size_t)dim_;
        }

        if (totalIlFloats > 0) {
            size_t ilBytes = totalIlFloats * sizeof(float);
            if (interleavedCodesBuf_ != nil) {
                resources_->deallocBuffer(
                        interleavedCodesBuf_, MetalAllocType::IVFLists);
                interleavedCodesBuf_ = nil;
            }
            interleavedCodesBuf_ =
                    resources_->allocBuffer(ilBytes, MetalAllocType::IVFLists);
            if (interleavedCodesBuf_) {
                auto* dst = reinterpret_cast<float*>([interleavedCodesBuf_ contents]);
                std::memset(dst, 0, ilBytes);

                for (size_t l = 0; l < (size_t)nlist_; ++l) {
                    size_t len = listLength_[l];
                    if (len == 0) continue;
                    size_t srcOff = listOffset_[l];
                    size_t dstOff = ilOffsets[l];
                    size_t numBlocks = (len + G - 1) / G;

                    for (size_t b = 0; b < numBlocks; ++b) {
                        for (int dd = 0; dd < dim_; ++dd) {
                            for (int g = 0; g < G; ++g) {
                                size_t vi = b * G + g;
                                float val = (vi < len)
                                        ? hostCodes_[(srcOff + vi) * (size_t)dim_ + dd]
                                        : 0.0f;
                                dst[dstOff + b * G * (size_t)dim_ + dd * G + g] = val;
                            }
                        }
                    }
                }
            }

            if (metaBytes > 0 && interleavedCodesBuf_) {
                if (interleavedCodesOffsetBuf_ != nil) {
                    resources_->deallocBuffer(
                            interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
                    interleavedCodesOffsetBuf_ = nil;
                }
                interleavedCodesOffsetBuf_ = resources_->allocBuffer(
                        metaBytes, MetalAllocType::IVFLists);
                if (interleavedCodesOffsetBuf_) {
                    auto* ptr = reinterpret_cast<uint32_t*>(
                            [interleavedCodesOffsetBuf_ contents]);
                    for (size_t i = 0; i < (size_t)nlist_; ++i) {
                        ptr[i] = ilOffsets[i];
                    }
                    haveInterleaved = true;
                }
            }
        }
    }

    // If interleaved layout is disabled, or interleaved allocation failed,
    // keep the canonical flat codes buffer so scan kernels can still run.
    if (!haveInterleaved) {
        if (interleavedCodesBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
            interleavedCodesBuf_ = nil;
        }
        if (interleavedCodesOffsetBuf_ != nil) {
            resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
            interleavedCodesOffsetBuf_ = nil;
        }
        if (codesBuffer_ == nil || [codesBuffer_ length] < codesBytes) {
            if (codesBuffer_ != nil) {
                resources_->deallocBuffer(codesBuffer_, MetalAllocType::IVFLists);
            }
            codesBuffer_ = resources_->allocBuffer(codesBytes, MetalAllocType::IVFLists);
            FAISS_THROW_IF_NOT_MSG(
                    codesBuffer_,
                    "MetalIVFFlatImpl: failed to allocate IVF codes buffer");
        }
        std::memcpy([codesBuffer_ contents], hostCodes_.data(), codesBytes);
    }
}

} // namespace gpu_metal
} // namespace faiss
