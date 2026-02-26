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

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu_metal {

MetalIVFFlatImpl::MetalIVFFlatImpl(
        std::shared_ptr<MetalResources> resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg)
        : resources_(std::move(resources)),
          dim_(dim),
          nlist_(nlist),
          metric_type_(metric),
          metric_arg_(metricArg),
          listLength_(nlist_, 0),
          listOffset_(nlist_, 0),
          totalVecs_(0),
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

    std::fill(listLength_.begin(), listLength_.end(), 0);
    std::fill(listOffset_.begin(), listOffset_.end(), 0);

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
    // Reserve on host; GPU buffers will be grown on next upload.
    hostCodes_.reserve(t * (size_t)dim_);
    hostIds_.reserve(t);
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

    // Compute new per-list lengths and offsets.
    std::vector<size_t> newLength(nlist_);
    std::vector<size_t> newOffset(nlist_);
    size_t prefix = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        newOffset[l] = prefix;
        newLength[l] = listLength_[l] + addPerList[l];
        prefix += newLength[l];
    }
    size_t newTotalVecs = prefix;

    // Allocate new host arrays and copy old data list by list.
    std::vector<float> newCodes(newTotalVecs * (size_t)dim_);
    std::vector<idx_t> newIds(newTotalVecs);

    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        size_t oldLen = listLength_[l];
        if (oldLen == 0) {
            continue;
        }
        size_t oldOff = listOffset_[l];
        size_t newOff = newOffset[l];
        std::memcpy(
                newCodes.data() + newOff * (size_t)dim_,
                hostCodes_.data() + oldOff * (size_t)dim_,
                oldLen * (size_t)dim_ * sizeof(float));
        std::memcpy(
                newIds.data() + newOff,
                hostIds_.data() + oldOff,
                oldLen * sizeof(idx_t));
    }

    // Append new data per list.
    std::vector<size_t> curPerList(nlist_);
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        curPerList[l] = listLength_[l];
    }

    for (idx_t i = 0; i < n; ++i) {
        idx_t list = list_nos[i];
        if (list < 0 || list >= nlist_) {
            continue;
        }
        size_t l = (size_t)list;
        size_t dstIndex = newOffset[l] + curPerList[l];
        curPerList[l]++;

        // Copy vector
        const float* xi = x + (size_t)i * (size_t)dim_;
        std::memcpy(
                newCodes.data() + dstIndex * (size_t)dim_,
                xi,
                (size_t)dim_ * sizeof(float));

        // Copy id
        idx_t id = xids ? xids[i] : (idx_t)(totalVecs_ + (size_t)(i));
        newIds[dstIndex] = id;
    }

    // Commit new host state.
    hostCodes_.swap(newCodes);
    hostIds_.swap(newIds);
    listLength_.swap(newLength);
    listOffset_.swap(newOffset);
    totalVecs_ = newTotalVecs;

    // Upload to GPU.
    uploadToGpu();
}

void MetalIVFFlatImpl::uploadToGpu() {
    if (!resources_ || !resources_->isAvailable()) {
        return;
    }

    size_t codesBytes = hostCodes_.size() * sizeof(float);
    size_t idsBytes   = hostIds_.size() * sizeof(idx_t);
    size_t metaBytes  = (size_t)nlist_ * sizeof(uint32_t);

    // Free and reallocate GPU buffers sized exactly for current data.
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

    // Always (re)upload metadata buffers — they reflect current list layout.
    if (metaBytes > 0) {
        listOffsetBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        listLengthBuf_ = resources_->allocBuffer(metaBytes, MetalAllocType::IVFLists);
        FAISS_THROW_IF_NOT_MSG(
                listOffsetBuf_ && listLengthBuf_,
                "MetalIVFFlatImpl: failed to allocate metadata GPU buffers");

        auto* offPtr = reinterpret_cast<uint32_t*>([listOffsetBuf_ contents]);
        auto* lenPtr = reinterpret_cast<uint32_t*>([listLengthBuf_ contents]);
        for (size_t i = 0; i < (size_t)nlist_; ++i) {
            offPtr[i] = (uint32_t)listOffset_[i];
            lenPtr[i] = (uint32_t)listLength_[i];
        }
    }

    if (codesBytes == 0 || idsBytes == 0) {
        return;
    }

    codesBuffer_ = resources_->allocBuffer(codesBytes, MetalAllocType::IVFLists);
    idsBuffer_   = resources_->allocBuffer(idsBytes,   MetalAllocType::IVFLists);

    FAISS_THROW_IF_NOT_MSG(
            codesBuffer_ && idsBuffer_,
            "MetalIVFFlatImpl: failed to allocate IVF Metal buffers");

    std::memcpy([codesBuffer_ contents], hostCodes_.data(), codesBytes);
    std::memcpy([idsBuffer_   contents], hostIds_.data(),   idsBytes);

    // Build interleaved codes buffer: blocks of 32 vectors with dims interleaved.
    // Layout per block: [v0d0 v1d0 ... v31d0] [v0d1 v1d1 ... v31d1] ...
    if (interleavedCodesBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesBuf_, MetalAllocType::IVFLists);
        interleavedCodesBuf_ = nil;
    }
    if (interleavedCodesOffsetBuf_ != nil) {
        resources_->deallocBuffer(interleavedCodesOffsetBuf_, MetalAllocType::IVFLists);
        interleavedCodesOffsetBuf_ = nil;
    }

    constexpr int G = kInterleavedGroupSize; // 32
    std::vector<uint32_t> ilOffsets((size_t)nlist_);
    size_t totalIlFloats = 0;
    for (size_t l = 0; l < (size_t)nlist_; ++l) {
        ilOffsets[l] = (uint32_t)totalIlFloats;
        size_t len = listLength_[l];
        size_t numBlocks = (len + G - 1) / G;
        totalIlFloats += numBlocks * G * (size_t)dim_;
    }

    if (totalIlFloats > 0) {
        size_t ilBytes = totalIlFloats * sizeof(float);
        interleavedCodesBuf_ = resources_->allocBuffer(ilBytes, MetalAllocType::IVFLists);
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
    }

    if (metaBytes > 0 && totalIlFloats > 0) {
        interleavedCodesOffsetBuf_ = resources_->allocBuffer(
                metaBytes, MetalAllocType::IVFLists);
        if (interleavedCodesOffsetBuf_) {
            auto* ptr = reinterpret_cast<uint32_t*>(
                    [interleavedCodesOffsetBuf_ contents]);
            for (size_t i = 0; i < (size_t)nlist_; ++i) {
                ptr[i] = ilOffsets[i];
            }
        }
    }
}

} // namespace gpu_metal
} // namespace faiss

