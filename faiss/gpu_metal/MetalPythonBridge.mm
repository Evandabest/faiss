// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalPythonBridge.h"
#import "MetalCloner.h"
#import "MetalDistance.h"
#import "StandardMetalResources.h"
#include <faiss/impl/FaissAssert.h>
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
    if (!res || !res->impl) {
        return nullptr;
    }
    return index_cpu_to_metal_gpu(
            static_cast<StandardMetalResources*>(res->impl),
            device, index, options);
}

faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::Index* index,
        const MetalClonerOptions* options) {
    if (res.size() != 1 || devices.size() != 1) {
        return nullptr;
    }
    return index_cpu_to_gpu(res[0], devices[0], index, options);
}

faiss::Index* index_gpu_to_cpu(const faiss::Index* index) {
    return index_metal_gpu_to_cpu(index);
}

void bfKnn(
        StandardMetalResourcesHolder* res,
        const MetalBridgeDistanceParams& args) {
    FAISS_THROW_IF_NOT_MSG(
            res && res->impl,
            "bfKnn (MetalPythonBridge): resources must be non-null");
    auto* impl = static_cast<StandardMetalResources*>(res->impl);
    ::faiss::gpu_metal::MetalDistanceParams coreArgs;
    coreArgs.metric = args.metric;
    coreArgs.metricArg = args.metricArg;
    coreArgs.k = args.k;
    coreArgs.dims = args.dims;
    coreArgs.vectors = args.vectors;
    coreArgs.vectorType = ::faiss::gpu_metal::MetalDistanceDataType::F32;
    coreArgs.vectorsRowMajor = args.vectorsRowMajor;
    coreArgs.numVectors = args.numVectors;
    coreArgs.vectorNorms = args.vectorNorms;
    coreArgs.queries = args.queries;
    coreArgs.queryType = ::faiss::gpu_metal::MetalDistanceDataType::F32;
    coreArgs.queriesRowMajor = args.queriesRowMajor;
    coreArgs.numQueries = args.numQueries;
    coreArgs.outDistances = args.outDistances;
    coreArgs.ignoreOutDistances = args.ignoreOutDistances;
    coreArgs.outIndicesType =
            (args.outIndicesType == MetalBridgeIndicesDataType::I32)
            ? ::faiss::gpu_metal::MetalIndicesDataType::I32
            : ::faiss::gpu_metal::MetalIndicesDataType::I64;
    coreArgs.outIndices = args.outIndices;
    coreArgs.device = args.device;
    coreArgs.use_cuvs = args.use_cuvs;
    ::faiss::gpu_metal::bfKnn(impl->getResources(), coreArgs);
}

void bfKnn_tiling(
        StandardMetalResourcesHolder* res,
        const MetalBridgeDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
    FAISS_THROW_IF_NOT_MSG(
            res && res->impl,
            "bfKnn_tiling (MetalPythonBridge): resources must be non-null");
    auto* impl = static_cast<StandardMetalResources*>(res->impl);
    ::faiss::gpu_metal::MetalDistanceParams coreArgs;
    coreArgs.metric = args.metric;
    coreArgs.metricArg = args.metricArg;
    coreArgs.k = args.k;
    coreArgs.dims = args.dims;
    coreArgs.vectors = args.vectors;
    coreArgs.vectorType = ::faiss::gpu_metal::MetalDistanceDataType::F32;
    coreArgs.vectorsRowMajor = args.vectorsRowMajor;
    coreArgs.numVectors = args.numVectors;
    coreArgs.vectorNorms = args.vectorNorms;
    coreArgs.queries = args.queries;
    coreArgs.queryType = ::faiss::gpu_metal::MetalDistanceDataType::F32;
    coreArgs.queriesRowMajor = args.queriesRowMajor;
    coreArgs.numQueries = args.numQueries;
    coreArgs.outDistances = args.outDistances;
    coreArgs.ignoreOutDistances = args.ignoreOutDistances;
    coreArgs.outIndicesType =
            (args.outIndicesType == MetalBridgeIndicesDataType::I32)
            ? ::faiss::gpu_metal::MetalIndicesDataType::I32
            : ::faiss::gpu_metal::MetalIndicesDataType::I64;
    coreArgs.outIndices = args.outIndices;
    coreArgs.device = args.device;
    coreArgs.use_cuvs = args.use_cuvs;
    ::faiss::gpu_metal::bfKnn_tiling(
            impl->getResources(),
            coreArgs,
            vectorsMemoryLimit,
            queriesMemoryLimit);
}

} // namespace gpu_metal
} // namespace faiss
