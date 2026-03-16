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
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <memory>

namespace faiss {
namespace gpu_metal {

namespace {

std::atomic<int> gMetalProfilerDepth{0};

::faiss::gpu_metal::MetalClonerOptions toCoreClonerOptions(
        const MetalBridgeClonerOptions* opts) {
    ::faiss::gpu_metal::MetalClonerOptions out;
    if (!opts) {
        return out;
    }
    out.indicesOptions = opts->indicesOptions;
    out.useFloat16 = opts->useFloat16;
    out.useFloat16CoarseQuantizer = opts->useFloat16CoarseQuantizer;
    out.reserveVecs = opts->reserveVecs;
    out.interleavedLayout = opts->interleavedLayout;
    out.useIVFScalarQuantizer = opts->useIVFScalarQuantizer;
    out.ivfSQType = opts->ivfSQType;
    out.storeTransposed = opts->storeTransposed;
    out.allowCpuCoarseQuantizer = opts->allowCpuCoarseQuantizer;
    out.verbose = opts->verbose;
    return out;
}

::faiss::gpu_metal::MetalDistanceDataType toCoreDistanceType(
        MetalBridgeDistanceDataType t) {
    switch (t) {
        case MetalBridgeDistanceDataType::F32:
            return ::faiss::gpu_metal::MetalDistanceDataType::F32;
        case MetalBridgeDistanceDataType::F16:
            return ::faiss::gpu_metal::MetalDistanceDataType::F16;
        case MetalBridgeDistanceDataType::BF16:
            return ::faiss::gpu_metal::MetalDistanceDataType::BF16;
    }
    return ::faiss::gpu_metal::MetalDistanceDataType::F32;
}

} // namespace

StandardMetalResourcesHolder::StandardMetalResourcesHolder() {
    impl = new StandardMetalResources();
}

StandardMetalResourcesHolder::~StandardMetalResourcesHolder() {
    delete static_cast<StandardMetalResources*>(impl);
    impl = nullptr;
}

// get_num_gpus() is implemented in MetalCloner.mm; bridge header declares it for SWIG.
void gpu_profiler_start() {
    const int depth = gMetalProfilerDepth.fetch_add(1) + 1;
    if (std::getenv("FAISS_METAL_PROFILE_LOG")) {
        std::fprintf(stderr, "[faiss_metal] gpu_profiler_start depth=%d\n", depth);
    }
}

void gpu_profiler_stop() {
    int depth = gMetalProfilerDepth.load();
    while (depth > 0 &&
           !gMetalProfilerDepth.compare_exchange_weak(depth, depth - 1)) {
    }
    if (std::getenv("FAISS_METAL_PROFILE_LOG")) {
        std::fprintf(
                stderr,
                "[faiss_metal] gpu_profiler_stop depth=%d\n",
                gMetalProfilerDepth.load());
    }
}

void gpu_sync_all_devices() {
    auto res = std::make_shared<MetalResources>();
    if (res && res->isAvailable()) {
        res->synchronize();
    }
}

faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index,
        const MetalBridgeClonerOptions* options) {
    if (!res || !res->impl) {
        return nullptr;
    }
    auto coreOpts = toCoreClonerOptions(options);
    return index_cpu_to_metal_gpu(
            static_cast<StandardMetalResources*>(res->impl),
            device, index, &coreOpts);
}

faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::Index* index,
        const MetalBridgeClonerOptions* options) {
    if (res.size() != 1 || devices.size() != 1) {
        return nullptr;
    }
    return index_cpu_to_gpu(res[0], devices[0], index, options);
}

faiss::Index* index_gpu_to_cpu(const faiss::Index* index) {
    return index_metal_gpu_to_cpu(index);
}

faiss::IndexBinary* index_binary_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::IndexBinary* index,
        const MetalBridgeClonerOptions* options) {
    (void)options;
    if (!res || !res->impl) {
        return nullptr;
    }
    return index_binary_cpu_to_metal_gpu(
            static_cast<StandardMetalResources*>(res->impl),
            device,
            index);
}

faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
        std::vector<StandardMetalResourcesHolder*>& res,
        std::vector<int>& devices,
        const faiss::IndexBinary* index,
        const MetalBridgeClonerOptions* options) {
    if (res.size() != 1 || devices.size() != 1) {
        return nullptr;
    }
    return index_binary_cpu_to_gpu(res[0], devices[0], index, options);
}

faiss::IndexBinary* index_binary_gpu_to_cpu(const faiss::IndexBinary* index) {
    return index_binary_metal_gpu_to_cpu(index);
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
    coreArgs.vectorType = toCoreDistanceType(args.vectorType);
    coreArgs.vectorsRowMajor = args.vectorsRowMajor;
    coreArgs.numVectors = static_cast<faiss::idx_t>(args.numVectors);
    coreArgs.vectorNorms = args.vectorNorms;
    coreArgs.queries = args.queries;
    coreArgs.queryType = toCoreDistanceType(args.queryType);
    coreArgs.queriesRowMajor = args.queriesRowMajor;
    coreArgs.numQueries = static_cast<faiss::idx_t>(args.numQueries);
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
    coreArgs.vectorType = toCoreDistanceType(args.vectorType);
    coreArgs.vectorsRowMajor = args.vectorsRowMajor;
    coreArgs.numVectors = static_cast<faiss::idx_t>(args.numVectors);
    coreArgs.vectorNorms = args.vectorNorms;
    coreArgs.queries = args.queries;
    coreArgs.queryType = toCoreDistanceType(args.queryType);
    coreArgs.queriesRowMajor = args.queriesRowMajor;
    coreArgs.numQueries = static_cast<faiss::idx_t>(args.numQueries);
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
