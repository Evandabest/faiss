// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * C++-only API for Python/SWIG. No Objective-C types so SWIG can parse it.
 * Implemented in MetalPythonBridge.mm which includes the real Metal backend.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <cstdint>
#include <vector>

namespace faiss {
namespace gpu_metal {

/// Opaque holder for Metal resources (SWIG sees this as StandardGpuResources).
struct StandardMetalResourcesHolder {
    void* impl = nullptr;
    StandardMetalResourcesHolder();
    ~StandardMetalResourcesHolder();
    StandardMetalResourcesHolder(const StandardMetalResourcesHolder&) = delete;
    StandardMetalResourcesHolder& operator=(const StandardMetalResourcesHolder&) = delete;
};

/// SWIG-facing cloner options mirror.
struct MetalBridgeClonerOptions {
    faiss::gpu::IndicesOptions indicesOptions = faiss::gpu::INDICES_64_BIT;
    bool useFloat16 = false;
    bool useFloat16CoarseQuantizer = false;
    long reserveVecs = 0;
    bool interleavedLayout = true;
    bool storeTransposed = false;
    bool allowCpuCoarseQuantizer = false;
    bool verbose = false;
};

/// Bridge enum for vector/query scalar type in distance API.
enum class MetalBridgeDistanceDataType {
    F32 = 1,
    F16,
    BF16,
};

/// Bridge enum for output index type in distance API.
enum class MetalBridgeIndicesDataType {
    I64 = 1,
    I32,
};

/// Bridge struct mirroring CUDA-style distance params for Python wrappers.
struct MetalBridgeDistanceParams {
    faiss::MetricType metric = METRIC_L2;
    float metricArg = 0.0f;
    int k = 0;
    int dims = 0;

    const void* vectors = nullptr;
    MetalBridgeDistanceDataType vectorType = MetalBridgeDistanceDataType::F32;
    bool vectorsRowMajor = true;
    int64_t numVectors = 0;
    const float* vectorNorms = nullptr;

    const void* queries = nullptr;
    MetalBridgeDistanceDataType queryType = MetalBridgeDistanceDataType::F32;
    bool queriesRowMajor = true;
    int64_t numQueries = 0;

    float* outDistances = nullptr;
    bool ignoreOutDistances = false;
    MetalBridgeIndicesDataType outIndicesType = MetalBridgeIndicesDataType::I64;
    void* outIndices = nullptr;

    int device = -1;
    bool use_cuvs = false;
};

/// Same names as GPU API for unified Python binding.
int get_num_gpus();
void gpu_profiler_start();
void gpu_profiler_stop();
void gpu_sync_all_devices();

/// Clone CPU index to Metal GPU. Caller owns returned index.
/// options supports a Metal-compatible subset of GpuClonerOptions.
faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index,
        const MetalBridgeClonerOptions* options = nullptr);

/// Multi-GPU clone: only single-device supported; calls index_cpu_to_gpu when size==1.
faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::Index* index,
        const MetalBridgeClonerOptions* options = nullptr);

/// Copy Metal index back to CPU. Caller owns returned index.
faiss::Index* index_gpu_to_cpu(const faiss::Index* index);

/// Clone CPU binary index to Metal GPU. Caller owns returned index.
faiss::IndexBinary* index_binary_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::IndexBinary* index,
        const MetalBridgeClonerOptions* options = nullptr);

/// Multi-GPU binary clone: single-device only for Metal.
faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::IndexBinary* index,
        const MetalBridgeClonerOptions* options = nullptr);

/// Copy Metal binary index back to CPU. Caller owns returned index.
faiss::IndexBinary* index_binary_gpu_to_cpu(
        const faiss::IndexBinary* index);

/// Distance API parity helpers for Python wrappers.
void bfKnn(
        StandardMetalResourcesHolder* res,
        const MetalBridgeDistanceParams& args);

void bfKnn_tiling(
        StandardMetalResourcesHolder* res,
        const MetalBridgeDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit);

} // namespace gpu_metal
} // namespace faiss
