// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexFlat.h"
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <cstring>

namespace faiss {
namespace gpu_metal {

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config),
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
    size_t newSize = newCap * (size_t)d * sizeof(float);
    id<MTLBuffer> newBuf =
            resources_->allocBuffer(newSize, MetalAllocType::FlatData);
    FAISS_THROW_IF_NOT_MSG(newBuf != nil, "MetalIndexFlat: failed to allocate buffer");
    if (ntotal > 0 && vectorsBuffer_ != nil) {
        std::memcpy(
                [newBuf contents],
                [vectorsBuffer_ contents],
                (size_t)ntotal * (size_t)d * sizeof(float));
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
    float* ptr = (float*)[vectorsBuffer_ contents];
    std::memcpy(ptr + (size_t)ntotal * (size_t)d, x, (size_t)n * (size_t)d * sizeof(float));
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
    float* ptr = (float*)[vectorsBuffer_ contents];
    std::memcpy(ptr + (size_t)ntotal * (size_t)d, x, (size_t)n * (size_t)d * sizeof(float));
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
    FAISS_THROW_IF_NOT(k > 0);
    if (ntotal == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
        }
        return;
    }
    IDSelector* sel = params ? params->sel : nullptr;
    std::vector<float> hostVectors((size_t)ntotal * (size_t)d);
    std::memcpy(
            hostVectors.data(),
            [vectorsBuffer_ contents],
            (size_t)ntotal * (size_t)d * sizeof(float));
    const float* y = hostVectors.data();

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, y, d, size_t(n), size_t(ntotal), &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, y, d, size_t(n), size_t(ntotal), &res, nullptr, sel);
    } else {
        FAISS_THROW_MSG("MetalIndexFlat: only L2 and inner product supported");
    }

    for (idx_t i = 0; i < n * k; ++i) {
        if (labels[i] >= 0 && labels[i] < ntotal) {
            labels[i] = ids_[labels[i]];
        }
    }
}

} // namespace gpu_metal
} // namespace faiss
