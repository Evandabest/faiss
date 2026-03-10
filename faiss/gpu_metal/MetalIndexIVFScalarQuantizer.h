// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF Scalar Quantizer wrapper — supports QT_8bit and QT_fp16.
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu_metal/MetalIndex.h>

#include <memory>

namespace faiss {
namespace gpu_metal {

class MetalIVFSQImpl;

class MetalIndexIVFScalarQuantizer : public MetalIndex {
public:
    MetalIndexIVFScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            int dims,
            idx_t nlist,
            faiss::ScalarQuantizer::QuantizerType sqType,
            faiss::MetricType metric = METRIC_L2,
            bool byResidual = true,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexIVFScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexIVFScalarQuantizer* cpuIndex,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexIVFScalarQuantizer() override;

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

    void copyFrom(const faiss::IndexIVFScalarQuantizer* index);
    void copyTo(faiss::IndexIVFScalarQuantizer* index) const;

    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void updateQuantizer();
    std::vector<idx_t> getListIndices(idx_t listId) const;
    void reclaimMemory();

    idx_t nlist() const;
    size_t nprobe() const;
    faiss::ScalarQuantizer::QuantizerType sqQuantizerType() const;

private:
    std::unique_ptr<faiss::IndexIVFScalarQuantizer> cpuIndex_;
    std::unique_ptr<MetalIVFSQImpl> gpuIvf_;

    mutable id<MTLBuffer> searchQueriesBuf_ = nil;
    mutable id<MTLBuffer> searchCoarseBuf_  = nil;
    mutable id<MTLBuffer> searchOutDistBuf_ = nil;
    mutable id<MTLBuffer> searchOutIdxBuf_  = nil;
    mutable size_t searchQueriesCap_ = 0;
    mutable size_t searchCoarseCap_  = 0;
    mutable size_t searchOutDistCap_ = 0;
    mutable size_t searchOutIdxCap_  = 0;
    mutable id<MTLBuffer> searchPerListDistBuf_ = nil;
    mutable id<MTLBuffer> searchPerListIdxBuf_  = nil;
    mutable size_t searchPerListDistCap_ = 0;
    mutable size_t searchPerListIdxCap_  = 0;

    mutable id<MTLBuffer> centroidBuf_      = nil;
    mutable id<MTLBuffer> centroidNormsBuf_ = nil;

    void ensureSearchBuf_(
            id<MTLBuffer>& buf,
            size_t& cap,
            size_t needed) const;

    void uploadCentroids_() const;

    /// Upload SQ decode tables to the GPU impl.
    void uploadSQTables_() const;
};

} // namespace gpu_metal
} // namespace faiss
