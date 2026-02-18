// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Reusable distance computation for Metal backend.
 * Mirrors faiss/gpu/impl/Distance.cu/Distance.cuh for CUDA.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>

namespace faiss {
namespace gpu_metal {

/// Calculate tile sizes for distance computation (mirrors CUDA's chooseTileSize).
/// Determines optimal query and vector tile dimensions based on available memory.
void chooseTileSize(
        int nq,
        int nb,
        int d,
        size_t elementSize,
        size_t availableMem,
        int& tileRows,
        int& tileCols);

/// Maximum k supported by Metal distance computation (2048; heap in threadgroup memory).
int getMetalDistanceMaxK();

/// Reusable distance computation function (mirrors CUDA's bfKnnOnDevice).
/// Computes brute-force k-NN distances between queries and vectors using Metal.
/// This is the internal function that can be reused by multiple index types.
///
/// @param device Metal device
/// @param queue Metal command queue
/// @param queries Query vectors (nq * d) float, row-major
/// @param vectors Database vectors (nb * d) float, row-major
/// @param nq Number of queries
/// @param nb Number of vectors
/// @param d Vector dimension
/// @param k Number of nearest neighbors to return
/// @param isL2 true = L2 squared distance, false = inner product
/// @param outDistances Output distances (nq * k) float
/// @param outIndices Output indices (nq * k) int32
/// @returns true on success, false on failure
bool runMetalDistance(
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
        id<MTLBuffer> outIndices);

/// L2 distance computation (convenience wrapper).
/// Computes L2 squared distance and returns top-k results.
bool runMetalL2Distance(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq,
        int nb,
        int d,
        int k,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices);

/// Inner product distance computation (convenience wrapper).
/// Computes inner product and returns top-k results.
bool runMetalIPDistance(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq,
        int nb,
        int d,
        int k,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices);

} // namespace gpu_metal
} // namespace faiss
