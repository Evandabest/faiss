// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * FlatIndex-specific wrapper for Metal distance computation.
 * Mirrors faiss/gpu/impl/FlatIndex.cuh for CUDA.
 * Delegates to MetalDistance.h for actual computation.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>

namespace faiss {
namespace gpu_metal {

/// FlatIndex-specific wrapper: runs GPU search for FlatIndex.
/// Delegates to runMetalDistance() from MetalDistance.h.
/// Maximum k supported by the GPU top-k kernel (2048; heap in threadgroup memory).
int getMetalFlatSearchMaxK();

/// Returns true on success; false if pipeline creation failed.
/// Thin wrapper that calls runMetalDistance() from MetalDistance.mm.
bool runFlatSearchGPU(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,      // (nq * d) float, row-major
        id<MTLBuffer> vectors,     // (nb * d) float, row-major
        int nq,
        int nb,
        int d,
        int k,
        bool isL2,                  // true = L2 squared, false = inner product
        id<MTLBuffer> outDistances, // (nq * k) float
        id<MTLBuffer> outIndices);  // (nq * k) int32

} // namespace gpu_metal
} // namespace faiss
