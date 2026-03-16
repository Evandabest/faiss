// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF Scalar Quantizer implementation: GPU-resident IVF list storage
 * for SQ-encoded codes.
 * GPU kernels currently consume SQ8 and FP16 code layouts.
 * Mirrors the roles of faiss/gpu/impl/IVFFlat.cuh (storage side only).
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/MetalResources.h>

namespace faiss {
namespace gpu_metal {

/// GPU-resident IVF list storage for scalar-quantized codes.
///
/// SQ8:  each vector is d uint8 bytes; decode tables (vmin, vdiff) stored
///       in a separate GPU buffer.
/// FP16: each vector is d half values; no decode tables needed.
class MetalIVFSQImpl {
public:
    MetalIVFSQImpl(
            std::shared_ptr<MetalResources> resources,
            int dim,
            idx_t nlist,
            MetalSQType sqType,
            faiss::MetricType metric,
            float metricArg);

    ~MetalIVFSQImpl();

    void reset();

    void reserveMemory(idx_t totalVecs);

    /// Append a batch of pre-encoded SQ codes to IVF lists.
    /// - codes: host pointer, n * code_size bytes (uint8 for SQ8, half for FP16)
    /// - list_nos: host pointer, size n; -1 entries skipped
    /// - xids: host pointer, size n (null → auto ids)
    void appendCodes(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            const idx_t* xids);

    /// Upload per-dimension SQ decode tables (SQ4/SQ6/SQ8 only).
    /// tables: host pointer, vmin[0..d-1] then vdiff[0..d-1] = 2*d floats
    void setSQTables(const float* tables);

    int dim() const { return dim_; }
    idx_t nlist() const { return nlist_; }
    MetalSQType sqType() const { return sqType_; }
    size_t codeSize() const { return codeSize_; }
    faiss::MetricType metricType() const { return metric_type_; }
    float metricArg() const { return metric_arg_; }

    const std::vector<size_t>& listLength() const { return listLength_; }
    const std::vector<size_t>& listOffset() const { return listOffset_; }

    id<MTLBuffer> codesBuffer() const { return codesBuffer_; }
    id<MTLBuffer> idsBuffer() const { return idsBuffer_; }
    id<MTLBuffer> listOffsetGpuBuffer() const { return listOffsetBuf_; }
    id<MTLBuffer> listLengthGpuBuffer() const { return listLengthBuf_; }
    id<MTLBuffer> sqTablesBuffer() const { return sqTablesBuf_; }
    size_t totalVecs() const { return totalVecs_; }

private:
    void uploadToGpu();

    std::shared_ptr<MetalResources> resources_;

    int dim_;
    idx_t nlist_;
    MetalSQType sqType_;
    size_t codeSize_; // bytes per vector
    faiss::MetricType metric_type_;
    float metric_arg_;

    std::vector<size_t> listLength_;
    std::vector<size_t> listOffset_;

    std::vector<uint8_t> hostCodes_;
    std::vector<idx_t> hostIds_;
    size_t totalVecs_;

    id<MTLBuffer> codesBuffer_;
    id<MTLBuffer> idsBuffer_;
    id<MTLBuffer> listOffsetBuf_;
    id<MTLBuffer> listLengthBuf_;
    id<MTLBuffer> sqTablesBuf_;
};

} // namespace gpu_metal
} // namespace faiss
