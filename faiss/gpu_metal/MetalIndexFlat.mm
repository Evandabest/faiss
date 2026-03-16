// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexFlat.h"
#import "MetalFlatKernels.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>

namespace faiss {
namespace gpu_metal {

static void floatToHalf(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        __fp16 h = (__fp16)src[i];
        std::memcpy(&dst[i], &h, sizeof(uint16_t));
    }
}

static void halfToFloat(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        __fp16 h;
        std::memcpy(&h, &src[i], sizeof(uint16_t));
        dst[i] = (float)h;
    }
}

static void copyF32RowMajorToTransposed(
        const float* srcRowMajor,
        float* dstTransposed,
        idx_t n,
        int d,
        idx_t startVec,
        size_t leadingDim) {
    for (idx_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            dstTransposed[(size_t)j * leadingDim + (size_t)(startVec + i)] =
                    srcRowMajor[(size_t)i * (size_t)d + (size_t)j];
        }
    }
}

static void copyF16RowMajorToTransposed(
        const float* srcRowMajor,
        uint16_t* dstTransposed,
        idx_t n,
        int d,
        idx_t startVec,
        size_t leadingDim) {
    for (idx_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            __fp16 h = (__fp16)srcRowMajor[(size_t)i * (size_t)d + (size_t)j];
            std::memcpy(
                    &dstTransposed[(size_t)j * leadingDim + (size_t)(startVec + i)],
                    &h,
                    sizeof(uint16_t));
        }
    }
}

static void copyF32TransposedToRowMajor(
        const float* srcTransposed,
        float* dstRowMajor,
        idx_t n,
        int d,
        idx_t startVec,
        size_t leadingDim) {
    for (idx_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            dstRowMajor[(size_t)i * (size_t)d + (size_t)j] =
                    srcTransposed[(size_t)j * leadingDim + (size_t)(startVec + i)];
        }
    }
}

static void copyF16TransposedToRowMajor(
        const uint16_t* srcTransposed,
        float* dstRowMajor,
        idx_t n,
        int d,
        idx_t startVec,
        size_t leadingDim) {
    for (idx_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            __fp16 h;
            std::memcpy(
                    &h,
                    &srcTransposed[(size_t)j * leadingDim + (size_t)(startVec + i)],
                    sizeof(uint16_t));
            dstRowMajor[(size_t)i * (size_t)d + (size_t)j] = (float)h;
        }
    }
}

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config),
          useFloat16_(config.useFloat16),
          storeTransposed_(config.storeTransposed),
          vectorsBuffer_(nil),
          capacityVecs_(0) {
    FAISS_THROW_IF_NOT(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);
}

MetalIndexFlat::~MetalIndexFlat() {
    if (vectorsBuffer_ != nil) {
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
        vectorsBuffer_ = nil;
    }
    capacityVecs_ = 0;
}

void MetalIndexFlat::ensureCapacity(idx_t newNtotal) {
    if (newNtotal <= (idx_t)capacityVecs_) {
        return;
    }
    size_t newCap = (capacityVecs_ == 0)
            ? (size_t)newNtotal
            : std::max((size_t)newNtotal, capacityVecs_ * 2);
    size_t newSize = newCap * (size_t)d * elementSize();
    id<MTLBuffer> newBuf =
            resources_->allocBuffer(newSize, MetalAllocType::FlatData);
    FAISS_THROW_IF_NOT_MSG(newBuf != nil, "MetalIndexFlat: failed to allocate buffer");
    if (ntotal > 0 && vectorsBuffer_ != nil) {
        if (!storeTransposed_) {
            std::memcpy(
                    [newBuf contents],
                    [vectorsBuffer_ contents],
                    (size_t)ntotal * (size_t)d * elementSize());
        } else {
            const size_t oldLd = capacityVecs_;
            const size_t newLd = newCap;
            if (useFloat16_) {
                const auto* src =
                        reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]);
                auto* dst = reinterpret_cast<uint16_t*>([newBuf contents]);
                for (int j = 0; j < d; ++j) {
                    std::memcpy(
                            dst + (size_t)j * newLd,
                            src + (size_t)j * oldLd,
                            (size_t)ntotal * sizeof(uint16_t));
                }
            } else {
                const auto* src =
                        reinterpret_cast<const float*>([vectorsBuffer_ contents]);
                auto* dst = reinterpret_cast<float*>([newBuf contents]);
                for (int j = 0; j < d; ++j) {
                    std::memcpy(
                            dst + (size_t)j * newLd,
                            src + (size_t)j * oldLd,
                            (size_t)ntotal * sizeof(float));
                }
            }
        }
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
    }
    vectorsBuffer_ = newBuf;
    capacityVecs_ = newCap;
}

void MetalIndexFlat::add(idx_t n, const float* x) {
    if (n == 0) {
        return;
    }
    ensureCapacity(ntotal + n);
    size_t count = (size_t)n * (size_t)d;
    if (useFloat16_) {
        auto* dst = reinterpret_cast<uint16_t*>([vectorsBuffer_ contents]);
        if (storeTransposed_) {
            copyF16RowMajorToTransposed(x, dst, n, d, ntotal, capacityVecs_);
        } else {
            size_t offset = (size_t)ntotal * (size_t)d;
            floatToHalf(x, dst + offset, count);
        }
    } else {
        auto* dst = reinterpret_cast<float*>([vectorsBuffer_ contents]);
        if (storeTransposed_) {
            copyF32RowMajorToTransposed(x, dst, n, d, ntotal, capacityVecs_);
        } else {
            size_t offset = (size_t)ntotal * (size_t)d;
            std::memcpy(dst + offset, x, count * sizeof(float));
        }
    }
    for (idx_t i = 0; i < n; ++i) {
        ids_.push_back(ntotal + i);
    }
    ntotal += n;
}

void MetalIndexFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(xids != nullptr);
    ensureCapacity(ntotal + n);
    size_t count = (size_t)n * (size_t)d;
    if (useFloat16_) {
        auto* dst = reinterpret_cast<uint16_t*>([vectorsBuffer_ contents]);
        if (storeTransposed_) {
            copyF16RowMajorToTransposed(x, dst, n, d, ntotal, capacityVecs_);
        } else {
            size_t offset = (size_t)ntotal * (size_t)d;
            floatToHalf(x, dst + offset, count);
        }
    } else {
        auto* dst = reinterpret_cast<float*>([vectorsBuffer_ contents]);
        if (storeTransposed_) {
            copyF32RowMajorToTransposed(x, dst, n, d, ntotal, capacityVecs_);
        } else {
            size_t offset = (size_t)ntotal * (size_t)d;
            std::memcpy(dst + offset, x, count * sizeof(float));
        }
    }
    for (idx_t i = 0; i < n; ++i) {
        ids_.push_back(xids[i]);
    }
    ntotal += n;
}

void MetalIndexFlat::reset() {
    if (vectorsBuffer_ != nil) {
        resources_->deallocBuffer(vectorsBuffer_, MetalAllocType::FlatData);
        vectorsBuffer_ = nil;
    }
    capacityVecs_ = 0;
    ids_.clear();
    ntotal = 0;
}

void MetalIndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    (void)params;
    FAISS_THROW_IF_NOT(k > 0);
    if (ntotal == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
        }
        return;
    }
    const int maxK = getMetalFlatSearchMaxK();
    FAISS_THROW_IF_NOT_MSG(
            k <= maxK,
            "MetalIndexFlat: k exceeds GPU limit (see getMetalFlatSearchMaxK())");

    id<MTLDevice> device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();
    if (!device || !queue) {
        FAISS_THROW_MSG("MetalIndexFlat: device or queue not available");
    }

    const size_t queryBytes = (size_t)n * (size_t)d * sizeof(float);
    const size_t outDistBytes = (size_t)n * (size_t)k * sizeof(float);
    const size_t outIdxBytes = (size_t)n * (size_t)k * sizeof(int32_t);

    id<MTLBuffer> queryBuf = resources_->allocBuffer(queryBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> outDistBuf = resources_->allocBuffer(outDistBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> outIdxBuf = resources_->allocBuffer(outIdxBytes, MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(queryBuf && outDistBuf && outIdxBuf, "MetalIndexFlat: failed to allocate temp buffers");

    std::memcpy([queryBuf contents], x, queryBytes);

    const bool isL2 = (metric_type == METRIC_L2);
    id<MTLBuffer> searchVectorsBuf = vectorsBuffer_;
    id<MTLBuffer> transposedScratchBuf = nil;
    if (storeTransposed_) {
        const size_t vecBytes = (size_t)ntotal * (size_t)d * elementSize();
        transposedScratchBuf = resources_->allocBuffer(
                vecBytes, MetalAllocType::TemporaryMemoryBuffer);
        FAISS_THROW_IF_NOT_MSG(
                transposedScratchBuf != nil,
                "MetalIndexFlat: failed to allocate transposed scratch buffer");
        if (useFloat16_) {
            const auto* src =
                    reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]);
            auto* dst = reinterpret_cast<uint16_t*>([transposedScratchBuf contents]);
            for (idx_t i = 0; i < ntotal; ++i) {
                for (int j = 0; j < d; ++j) {
                    dst[(size_t)i * (size_t)d + (size_t)j] =
                            src[(size_t)j * capacityVecs_ + (size_t)i];
                }
            }
        } else {
            const auto* src =
                    reinterpret_cast<const float*>([vectorsBuffer_ contents]);
            auto* dst = reinterpret_cast<float*>([transposedScratchBuf contents]);
            for (idx_t i = 0; i < ntotal; ++i) {
                for (int j = 0; j < d; ++j) {
                    dst[(size_t)i * (size_t)d + (size_t)j] =
                            src[(size_t)j * capacityVecs_ + (size_t)i];
                }
            }
        }
        searchVectorsBuf = transposedScratchBuf;
    }

    bool ok;
    if (useFloat16_) {
        ok = runFlatSearchGPUFP16(
                device, queue, queryBuf, searchVectorsBuf,
                (int)n, (int)ntotal, d, (int)k, isL2,
                outDistBuf, outIdxBuf, resources_);
    } else {
        ok = runFlatSearchGPU(
                device, queue, queryBuf, searchVectorsBuf,
                (int)n, (int)ntotal, d, (int)k, isL2,
                outDistBuf, outIdxBuf, resources_);
    }

    resources_->deallocBuffer(queryBuf, MetalAllocType::TemporaryMemoryBuffer);
    if (transposedScratchBuf != nil) {
        resources_->deallocBuffer(
                transposedScratchBuf, MetalAllocType::TemporaryMemoryBuffer);
    }
    if (!ok) {
        resources_->deallocBuffer(outDistBuf, MetalAllocType::TemporaryMemoryBuffer);
        resources_->deallocBuffer(outIdxBuf, MetalAllocType::TemporaryMemoryBuffer);
        FAISS_THROW_MSG("MetalIndexFlat: GPU search failed (pipeline or dispatch error)");
    }

    std::memcpy(distances, [outDistBuf contents], outDistBytes);
    const int32_t* idxPtr = (const int32_t*)[outIdxBuf contents];
    for (idx_t i = 0; i < n * k; ++i) {
        int32_t idx = idxPtr[i];
        labels[i] = (idx >= 0 && idx < (int32_t)ntotal) ? ids_[idx] : -1;
    }

    resources_->deallocBuffer(outDistBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources_->deallocBuffer(outIdxBuf, MetalAllocType::TemporaryMemoryBuffer);
}

void MetalIndexFlat::copyTo(faiss::IndexFlat* index) const {
    FAISS_THROW_IF_NOT(index != nullptr);
    FAISS_THROW_IF_NOT(index->d == d);
    FAISS_THROW_IF_NOT(index->metric_type == metric_type);
    if (ntotal == 0 || vectorsBuffer_ == nil) {
        return;
    }
    size_t count = (size_t)ntotal * (size_t)d;
    std::vector<float> host(count);
    if (storeTransposed_) {
        if (useFloat16_) {
            copyF16TransposedToRowMajor(
                    reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]),
                    host.data(),
                    ntotal,
                    d,
                    0,
                    capacityVecs_);
        } else {
            copyF32TransposedToRowMajor(
                    reinterpret_cast<const float*>([vectorsBuffer_ contents]),
                    host.data(),
                    ntotal,
                    d,
                    0,
                    capacityVecs_);
        }
    } else if (useFloat16_) {
        halfToFloat(
                reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]),
                host.data(),
                count);
    } else {
        std::memcpy(host.data(), [vectorsBuffer_ contents], count * sizeof(float));
    }
    index->add(ntotal, host.data());
}

void MetalIndexFlat::copyFrom(const faiss::IndexFlat* index) {
    FAISS_THROW_IF_NOT(index != nullptr);
    FAISS_THROW_IF_NOT(index->d == d);
    FAISS_THROW_IF_NOT(index->metric_type == metric_type);

    reset();

    if (index->ntotal == 0) {
        return;
    }

    const idx_t n = index->ntotal;
    ensureCapacity(n);

    const float* src = index->get_xb();
    size_t count = (size_t)n * (size_t)d;
    if (storeTransposed_) {
        if (useFloat16_) {
            copyF16RowMajorToTransposed(
                    src,
                    reinterpret_cast<uint16_t*>([vectorsBuffer_ contents]),
                    n,
                    d,
                    0,
                    capacityVecs_);
        } else {
            copyF32RowMajorToTransposed(
                    src,
                    reinterpret_cast<float*>([vectorsBuffer_ contents]),
                    n,
                    d,
                    0,
                    capacityVecs_);
        }
    } else if (useFloat16_) {
        floatToHalf(
                src,
                reinterpret_cast<uint16_t*>([vectorsBuffer_ contents]),
                count);
    } else {
        std::memcpy([vectorsBuffer_ contents], src, count * sizeof(float));
    }

    ids_.resize(n);
    for (idx_t i = 0; i < n; ++i) {
        ids_[i] = i;
    }
    ntotal = n;
}

void MetalIndexFlat::assign(
        idx_t n, const float* x, idx_t* labels, idx_t k) const {
    FAISS_THROW_IF_NOT(k > 0);
    std::vector<float> distances(n * k);
    search(n, x, k, distances.data(), labels);
}

void MetalIndexFlat::compute_residual(
        const float* x, float* residual, idx_t key) const {
    compute_residual_n(1, x, residual, &key);
}

void MetalIndexFlat::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    for (idx_t i = 0; i < n; ++i) {
        std::vector<float> recons(d);
        reconstruct(keys[i], recons.data());
        const float* x = xs + i * d;
        float* r = residuals + i * d;
        for (int j = 0; j < d; ++j)
            r[j] = x[j] - recons[j];
    }
}

void MetalIndexFlat::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

void MetalIndexFlat::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT_FMT(
            i0 >= 0 && i0 + ni <= ntotal,
            "MetalIndexFlat::reconstruct_n: range [%zd, %zd) out of bounds (ntotal=%zd)",
            (size_t)i0, (size_t)(i0 + ni), (size_t)ntotal);
    if (ni == 0) return;
    FAISS_THROW_IF_NOT(vectorsBuffer_ != nil);
    if (storeTransposed_) {
        if (useFloat16_) {
            copyF16TransposedToRowMajor(
                    reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]),
                    recons,
                    ni,
                    d,
                    i0,
                    capacityVecs_);
        } else {
            copyF32TransposedToRowMajor(
                    reinterpret_cast<const float*>([vectorsBuffer_ contents]),
                    recons,
                    ni,
                    d,
                    i0,
                    capacityVecs_);
        }
    } else {
        size_t count = (size_t)ni * (size_t)d;
        size_t offset = (size_t)i0 * (size_t)d;
        if (useFloat16_) {
            const auto* src =
                    reinterpret_cast<const uint16_t*>([vectorsBuffer_ contents]);
            halfToFloat(src + offset, recons, count);
        } else {
            const auto* src =
                    reinterpret_cast<const float*>([vectorsBuffer_ contents]);
            std::memcpy(recons, src + offset, count * sizeof(float));
        }
    }
}

void MetalIndexFlat::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* recons) const {
    FAISS_THROW_IF_NOT(keys != nullptr);
    FAISS_THROW_IF_NOT(recons != nullptr);
    for (idx_t i = 0; i < n; ++i) {
        reconstruct(keys[i], recons + (size_t)i * (size_t)d);
    }
}

size_t MetalIndexFlat::sa_code_size() const {
    return (size_t)d * sizeof(float);
}

void MetalIndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n <= 0) {
        return;
    }
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(bytes != nullptr);
    std::memcpy(bytes, x, (size_t)n * (size_t)d * sizeof(float));
}

void MetalIndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n <= 0) {
        return;
    }
    FAISS_THROW_IF_NOT(bytes != nullptr);
    FAISS_THROW_IF_NOT(x != nullptr);
    std::memcpy(x, bytes, (size_t)n * (size_t)d * sizeof(float));
}

} // namespace gpu_metal
} // namespace faiss
