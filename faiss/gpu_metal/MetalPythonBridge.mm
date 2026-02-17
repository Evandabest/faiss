// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalPythonBridge.h"
#import "MetalCloner.h"
#import "StandardMetalResources.h"
#include <memory>

namespace faiss {
namespace gpu_metal {

StandardMetalResourcesHolder::StandardMetalResourcesHolder() {
    impl = new StandardMetalResources();
}

StandardMetalResourcesHolder::~StandardMetalResourcesHolder() {
    delete static_cast<StandardMetalResources*>(impl);
    impl = nullptr;
}

// get_num_gpus() is implemented in MetalCloner.mm; bridge header declares it for SWIG.

faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options) {
    (void)options;
    if (!res || !res->impl) {
        return nullptr;
    }
    return index_cpu_to_metal_gpu(static_cast<StandardMetalResources*>(res->impl), device, index);
}

faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::Index* index,
        const MetalClonerOptions* options) {
    (void)options;
    if (res.size() != 1 || devices.size() != 1) {
        return nullptr;  // Multi-GPU not supported
    }
    return index_cpu_to_gpu(res[0], devices[0], index);
}

faiss::Index* index_gpu_to_cpu(const faiss::Index* index) {
    return index_metal_gpu_to_cpu(index);
}

} // namespace gpu_metal
} // namespace faiss
