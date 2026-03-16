// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * FlatIndex-specific wrapper for Metal distance computation.
 * Mirrors faiss/gpu/impl/FlatIndex.cu for CUDA.
 * Delegates to MetalDistance.mm for actual computation.
 */

#import "MetalFlatKernels.h"
#import "MetalDistance.h"

namespace faiss {
namespace gpu_metal {

// Thin wrapper: delegates to reusable runMetalDistance() from MetalDistance.mm
bool runFlatSearchGPU(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq,
        int nb,
        int d,
        int k,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources) {
    return runMetalDistance(
            device,
            queue,
            queries,
            vectors,
            nq,
            nb,
            d,
            k,
            isL2,
            outDistances,
            outIndices,
            resources);
}

bool runFlatSearchGPUFP16(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq,
        int nb,
        int d,
        int k,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources) {
    return runMetalDistanceFP16(
            device,
            queue,
            queries,
            vectors,
            nq,
            nb,
            d,
            k,
            isL2,
            outDistances,
            outIndices,
            resources);
}

int getMetalFlatSearchMaxK() {
    return getMetalDistanceMaxK();
}

} // namespace gpu_metal
} // namespace faiss
