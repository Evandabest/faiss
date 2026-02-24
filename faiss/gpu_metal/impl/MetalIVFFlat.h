// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF Flat implementation: GPU-resident IVF list storage and helpers.
 * Mirrors the roles of faiss/gpu/impl/IVFFlat.cuh (storage side only).
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/Index.h>
#include <faiss/gpu_metal/MetalResources.h>

namespace faiss {
namespace gpu_metal {

/// GPU-resident IVF list storage for flat (float32) codes.
/// Layout: all lists are stored contiguously in a single codes/ids buffer;
/// lists are described by (listOffset[list], listLength[list]).
class MetalIVFFlatImpl {
public:
    MetalIVFFlatImpl(
            std::shared_ptr<MetalResources> resources,
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg);

    ~MetalIVFFlatImpl();

    /// Reset all IVF lists and free GPU storage.
    void reset();

    /// Reserve host/GPU storage for at least totalVecs vectors.
    void reserveMemory(idx_t totalVecs);

    /// Append a batch of vectors to IVF lists.
    /// - x: host pointer, size n * dim
    /// - list_nos: host pointer, size n; -1 entries are skipped
    /// - xids: host pointer, size n (may be null to use internal ids)
    void appendVectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            const idx_t* xids);

    /// Accessors for future GPU search path.
    int dim() const {
        return dim_;
    }
    idx_t nlist() const {
        return nlist_;
    }
    faiss::MetricType metricType() const {
        return metric_type_;
    }
    float metricArg() const {
        return metric_arg_;
    }

    const std::vector<size_t>& listLength() const {
        return listLength_;
    }
    const std::vector<size_t>& listOffset() const {
        return listOffset_;
    }

    id<MTLBuffer> codesBuffer() const {
        return codesBuffer_;
    }
    id<MTLBuffer> idsBuffer() const {
        return idsBuffer_;
    }
    /// Pre-built GPU buffer of (nlist) uint32_t offsets (updated on every add).
    id<MTLBuffer> listOffsetGpuBuffer() const {
        return listOffsetBuf_;
    }
    /// Pre-built GPU buffer of (nlist) uint32_t lengths (updated on every add).
    id<MTLBuffer> listLengthGpuBuffer() const {
        return listLengthBuf_;
    }

    size_t totalVecs() const {
        return totalVecs_;
    }

private:
    void uploadToGpu();

    std::shared_ptr<MetalResources> resources_;

    int dim_;
    idx_t nlist_;
    faiss::MetricType metric_type_;
    float metric_arg_;

    // Per-list metadata
    std::vector<size_t> listLength_;
    std::vector<size_t> listOffset_;

    // Host copies of IVF data (flat layout)
    std::vector<float> hostCodes_; // size = totalVecs_ * dim_
    std::vector<idx_t> hostIds_;   // size = totalVecs_
    size_t totalVecs_;

    // GPU storage
    id<MTLBuffer> codesBuffer_;
    id<MTLBuffer> idsBuffer_;
    id<MTLBuffer> listOffsetBuf_;  // (nlist) uint32_t, list element offsets
    id<MTLBuffer> listLengthBuf_;  // (nlist) uint32_t, list sizes
};

} // namespace gpu_metal
} // namespace faiss

