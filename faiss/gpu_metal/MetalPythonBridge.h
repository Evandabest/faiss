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
#include <faiss/gpu_metal/MetalCloner.h>
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

/// Same names as GPU API for unified Python binding.
int get_num_gpus();

/// Clone CPU index to Metal GPU. Caller owns returned index.
/// options is ignored but accepted for API compatibility with GPU cloner.
faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options = nullptr);

/// Multi-GPU clone: only single-device supported; calls index_cpu_to_gpu when size==1.
faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::Index* index,
        const MetalClonerOptions* options = nullptr);

/// Copy Metal index back to CPU. Caller owns returned index.
faiss::Index* index_gpu_to_cpu(const faiss::Index* index);

} // namespace gpu_metal
} // namespace faiss
