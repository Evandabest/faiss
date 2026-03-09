// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexBinaryFlat.h"

#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu_metal/MetalDistance.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace faiss {
namespace gpu_metal {

MetalIndexBinaryFlat::MetalIndexBinaryFlat(
        std::shared_ptr<MetalResources> resources,
        int d)
        : IndexBinary(d), resources_(std::move(resources)) {
    FAISS_THROW_IF_NOT(d > 0);
    FAISS_THROW_IF_NOT(d % 8 == 0);
    is_trained = true;
}

MetalIndexBinaryFlat::MetalIndexBinaryFlat(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexBinaryFlat* cpuIndex)
        : IndexBinary(cpuIndex->d), resources_(std::move(resources)) {
    FAISS_THROW_IF_NOT(cpuIndex->d > 0);
    FAISS_THROW_IF_NOT(cpuIndex->d % 8 == 0);
    is_trained = true;
    copyFrom(cpuIndex);
}

MetalIndexBinaryFlat::~MetalIndexBinaryFlat() = default;

void MetalIndexBinaryFlat::ensureBuf_(
        id<MTLBuffer>& buf,
        size_t& cap,
        size_t needed) const {
    if (buf != nil && cap >= needed) return;
    size_t newCap = std::max(needed, cap * 2);
    id<MTLDevice> device = resources_->getDevice();
    buf = [device newBufferWithLength:newCap
                              options:MTLResourceStorageModeShared];
    cap = buf ? newCap : 0;
}

void MetalIndexBinaryFlat::uploadToGpu() {
    if (hostVectors_.empty()) {
        vectorsBuf_ = nil;
        return;
    }
    id<MTLDevice> device = resources_->getDevice();
    if (!device) return;
    size_t bytes = hostVectors_.size();
    vectorsBuf_ = [device newBufferWithLength:bytes
                                      options:MTLResourceStorageModeShared];
    if (vectorsBuf_)
        std::memcpy([vectorsBuf_ contents], hostVectors_.data(), bytes);
}

void MetalIndexBinaryFlat::add(idx_t n, const uint8_t* x) {
    if (n == 0) return;
    FAISS_THROW_IF_NOT(x != nullptr);
    size_t cs = (size_t)code_size;
    size_t oldSize = hostVectors_.size();
    hostVectors_.resize(oldSize + (size_t)n * cs);
    std::memcpy(hostVectors_.data() + oldSize, x, (size_t)n * cs);
    ntotal += n;
    uploadToGpu();
}

void MetalIndexBinaryFlat::reset() {
    hostVectors_.clear();
    ntotal = 0;
    vectorsBuf_ = nil;
}

void MetalIndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters*) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            distances[i] = std::numeric_limits<int32_t>::max();
            labels[i] = -1;
        }
        return;
    }

    id<MTLDevice> device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();

    if (!device || !queue || !vectorsBuf_) {
        FAISS_THROW_MSG("MetalIndexBinaryFlat: GPU not available for search");
    }

    size_t cs = (size_t)code_size;
    size_t queryBytes = (size_t)n * cs;
    size_t distBytes = (size_t)n * (size_t)k * sizeof(int32_t);
    size_t idxBytes = (size_t)n * (size_t)k * sizeof(int64_t);

    ensureBuf_(queryBuf_, queryBufCap_, queryBytes);
    ensureBuf_(outDistBuf_, outDistBufCap_, distBytes);
    ensureBuf_(outIdxBuf_, outIdxBufCap_, idxBytes);

    if (!queryBuf_ || !outDistBuf_ || !outIdxBuf_) {
        FAISS_THROW_MSG("MetalIndexBinaryFlat: failed to allocate search buffers");
    }

    std::memcpy([queryBuf_ contents], x, queryBytes);

    bool ok = runMetalHammingDistance(
            device, queue,
            queryBuf_, vectorsBuf_,
            (int)n, (int)ntotal, (int)cs, (int)k,
            outDistBuf_, outIdxBuf_);

    if (!ok) {
        FAISS_THROW_MSG("MetalIndexBinaryFlat: GPU Hamming search failed");
    }

    const int32_t* distPtr = reinterpret_cast<const int32_t*>(
            [outDistBuf_ contents]);
    const int64_t* idxPtr = reinterpret_cast<const int64_t*>(
            [outIdxBuf_ contents]);
    for (idx_t i = 0; i < n * k; ++i) {
        distances[i] = distPtr[i];
        labels[i] = (idxPtr[i] < 0) ? -1 : (idx_t)idxPtr[i];
    }
}

void MetalIndexBinaryFlat::reconstruct(idx_t key, uint8_t* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    size_t cs = (size_t)code_size;
    std::memcpy(recons, hostVectors_.data() + (size_t)key * cs, cs);
}

void MetalIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat* src) {
    FAISS_THROW_IF_NOT(src != nullptr);
    d = src->d;
    code_size = src->code_size;
    ntotal = src->ntotal;
    is_trained = src->is_trained;
    metric_type = src->metric_type;

    hostVectors_.clear();
    if (src->ntotal > 0) {
        size_t bytes = (size_t)src->ntotal * (size_t)src->code_size;
        hostVectors_.resize(bytes);
        std::memcpy(hostVectors_.data(), src->xb.data(), bytes);
    }
    uploadToGpu();
}

void MetalIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat* dst) const {
    FAISS_THROW_IF_NOT(dst != nullptr);
    dst->d = d;
    dst->code_size = code_size;
    dst->ntotal = ntotal;
    dst->is_trained = is_trained;
    dst->metric_type = metric_type;

    dst->xb.resize((size_t)ntotal * (size_t)code_size);
    if (ntotal > 0)
        std::memcpy(dst->xb.data(), hostVectors_.data(), hostVectors_.size());
}

} // namespace gpu_metal
} // namespace faiss
