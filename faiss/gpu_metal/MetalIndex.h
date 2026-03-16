// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Objective-C++ header (uses MetalResources).
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Configuration for Metal index (mirrors GpuIndexConfig roles).
struct MetalIndexConfig {
    int device = 0;

    /// Store vectors as float16 (half) instead of float32.
    /// Halves GPU memory usage at the cost of reduced precision.
    /// Only affects MetalIndexFlat vector storage; queries remain float32.
    bool useFloat16 = false;

    /// Store IVF coarse-quantizer centroids as float16.
    /// Reduces centroid buffer memory; coarse assignment runs through
    /// fp16 distance kernels. Applies to MetalIndexIVFFlat and
    /// MetalIndexIVFScalarQuantizer.
    bool useFloat16CoarseQuantizer = false;

    /// How IVF labels are represented (mirrors GPU IndicesOptions).
    faiss::gpu::IndicesOptions indicesOptions = faiss::gpu::INDICES_64_BIT;

    /// Whether IVF list data should use interleaved layout for scan kernels.
    bool interleavedLayout = true;

    /// Store Flat vectors in transposed layout (d x n) for API parity.
    bool storeTransposed = false;
};

/// Base class for Metal-backed indexes. Mirrors faiss::gpu::GpuIndex.
class MetalIndex : public faiss::Index {
public:
    MetalIndex(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            MetalIndexConfig config = MetalIndexConfig());

    int getDevice() const { return config_.device; }
    std::shared_ptr<MetalResources> getResources() { return resources_; }
    std::shared_ptr<const MetalResources> getResources() const {
        return resources_;
    }

protected:
    std::shared_ptr<MetalResources> resources_;
    MetalIndexConfig config_;
};

} // namespace gpu_metal
} // namespace faiss
