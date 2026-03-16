// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Objective-C++ header (uses Metal types).
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/Index.h>
#include <faiss/gpu_metal/MetalIndex.h>

namespace faiss {
struct IndexFlat;
}
#include <memory>
#include <vector>

namespace faiss {
namespace gpu_metal {

/// Flat index that stores vectors in an MTLBuffer. Supports L2 and inner
/// product. Search runs on GPU via Metal compute (distance + top-k kernels).
class MetalIndexFlat : public MetalIndex {
public:
    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg = 0.0f,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexFlat() override;

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

    /// Copy vectors to a CPU IndexFlat (e.g. for index_metal_gpu_to_cpu).
    void copyTo(::faiss::IndexFlat* index) const;

    /// Copy vectors from a CPU IndexFlat into this Metal index, replacing
    /// any existing data (e.g. for index_cpu_to_metal_gpu).
    void copyFrom(const ::faiss::IndexFlat* index);

    /// Assign vectors to their nearest neighbor (returns labels only).
    void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const override;

    /// Compute residual: residual = x - reconstruct(key).
    void compute_residual(const float* x, float* residual, idx_t key)
            const override;

    /// Batch compute residuals: residuals[i] = xs[i] - reconstruct(keys[i]).
    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const override;

    /// Reconstruct a single stored vector by internal key (0-based).
    void reconstruct(idx_t key, float* recons) const override;

    /// Reconstruct n contiguous stored vectors starting at i0.
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /// Whether this index stores vectors as float16 (half).
    bool useFloat16() const { return useFloat16_; }
    bool storeTransposed() const { return storeTransposed_; }

private:
    void ensureCapacity(idx_t newNtotal);

    /// Bytes per stored vector component (2 for half, 4 for float).
    size_t elementSize() const { return useFloat16_ ? 2 : 4; }

    bool useFloat16_;
    bool storeTransposed_;

    /// Vector storage (row-major). float32 or half depending on useFloat16_.
    id<MTLBuffer> vectorsBuffer_;
    size_t capacityVecs_;
    std::vector<idx_t> ids_;
};

} // namespace gpu_metal
} // namespace faiss
