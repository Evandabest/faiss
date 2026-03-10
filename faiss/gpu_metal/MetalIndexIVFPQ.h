// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF-PQ index: 8-bit product quantization with precomputed
 * per-query lookup tables and GPU IVF list scanning.
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalIndex.h>

#include <memory>

namespace faiss {
namespace gpu_metal {
class MetalIVFPQImpl;
} // namespace gpu_metal
} // namespace faiss

namespace faiss {
namespace gpu_metal {

class MetalIndexIVFPQ : public MetalIndex {
public:
    MetalIndexIVFPQ(
            std::shared_ptr<MetalResources> resources,
            int dims,
            idx_t nlist,
            int M,
            int nbitsPerIdx,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexIVFPQ(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexIVFPQ* cpuIndex,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexIVFPQ() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void copyFrom(const faiss::IndexIVFPQ* index);
    void copyTo(faiss::IndexIVFPQ* index) const;

    void updateQuantizer();
    std::vector<idx_t> getListIndices(idx_t listId) const;
    void reclaimMemory();

    /// Pre-allocate GPU storage for the given total number of vectors.
    void reserveMemory(idx_t numVecs);

    idx_t nlist() const;
    size_t nprobe() const;
    int getNumSubQuantizers() const;

private:
    std::unique_ptr<faiss::IndexIVFPQ> cpuIndex_;
    std::unique_ptr<MetalIVFPQImpl> gpuIvf_;

    mutable id<MTLBuffer> searchQueriesBuf_ = nil;
    mutable id<MTLBuffer> searchCoarseBuf_  = nil;
    mutable id<MTLBuffer> searchOutDistBuf_ = nil;
    mutable id<MTLBuffer> searchOutIdxBuf_  = nil;
    mutable id<MTLBuffer> searchPerListDistBuf_ = nil;
    mutable id<MTLBuffer> searchPerListIdxBuf_  = nil;
    mutable id<MTLBuffer> lookupTableBuf_   = nil;
    mutable size_t searchQueriesCap_ = 0;
    mutable size_t searchCoarseCap_  = 0;
    mutable size_t searchOutDistCap_ = 0;
    mutable size_t searchOutIdxCap_  = 0;
    mutable size_t searchPerListDistCap_ = 0;
    mutable size_t searchPerListIdxCap_  = 0;
    mutable size_t lookupTableCap_   = 0;

    mutable id<MTLBuffer> centroidBuf_      = nil;
    mutable id<MTLBuffer> centroidNormsBuf_ = nil;

    void ensureSearchBuf_(
            id<MTLBuffer>& buf,
            size_t& cap,
            size_t needed) const;

    void uploadCentroids_() const;
    void uploadPQCentroids_() const;

    /// Compute per-(query, probe) PQ lookup tables on CPU.
    /// Output: (nq * nprobe * M * 256) floats.
    void computeLookupTables_(
            int nq,
            const float* queries,
            int nprobe,
            const idx_t* coarseAssign,
            float* tables) const;
};

} // namespace gpu_metal
} // namespace faiss
