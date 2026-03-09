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

#include <faiss/MetricType.h>
#include <cstddef>
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Calculate tile sizes for distance computation (mirrors CUDA's chooseTileSize).
/// Determines optimal query and vector tile dimensions. \p availableMem is the
/// byte budget for the tile working set (e.g. from system available memory).
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

/// Float16 variant: vectors buffer contains half-precision data.
/// Queries remain float32. Computation is float32; only storage is fp16.
bool runMetalDistanceFP16(
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

/// @param device       Metal device
/// @param queue        Metal command queue
/// @param queries      (nq * d) float, row-major GPU buffer
/// @param codes        GPU codes buffer (totalVecs * d float, list-contiguous)
/// @param ids          GPU ids buffer  (totalVecs int64 stored as int32 pairs)
/// @param listOffset   GPU buffer (nlist uint32): byte/element offset into codes
/// @param listLength   GPU buffer (nlist uint32): number of vectors per list
/// @param coarseAssign GPU buffer (nq * nprobe int32): coarse assignment list ids
/// @param nq           Number of queries
/// @param d            Vector dimension
/// @param k            Number of nearest neighbors
/// @param nprobe       Number of probed lists per query
/// @param isL2         true = L2 squared, false = inner product
/// @param outDistances    Output distances (nq * k float)
/// @param outIndices      Output indices   (nq * k int32)
/// @param perListDistBuf  Scratch buffer (nq*nprobe*k float), caller-owned
/// @param perListIdxBuf   Scratch buffer (nq*nprobe*k int32), caller-owned
/// @returns true on success
bool runMetalIVFFlatScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq,
        int d,
        int k,
        int nprobe,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> interleavedCodes = nil,
        id<MTLBuffer> interleavedCodesOffset = nil);

/// SQ quantizer type for IVF SQ scan.
enum class MetalSQType { SQ8, FP16 };

/// IVF Scalar Quantizer scan: scans SQ-encoded inverted lists on the GPU.
///
/// @param sqType       Quantizer type (SQ8 or FP16)
/// @param sqTables     SQ decode tables buffer (SQ8: vmin[d] + vdiff[d] = 2*d
///                     floats; nil for FP16)
bool runMetalIVFSQScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq,
        int d,
        int k,
        int nprobe,
        bool isL2,
        MetalSQType sqType,
        id<MTLBuffer> sqTables,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf);

/// IVF PQ scan: scans 8-bit PQ-encoded inverted lists on the GPU.
///
/// @param lookupTable  Per-(query, probe) distance lookup tables,
///                     layout (nq * nprobe * M * 256) floats.
///                     table[m * 256 + code] = partial distance for
///                     subquantizer m, code value `code`.
/// @param M            Number of sub-quantizers (code_size in bytes).
bool runMetalIVFPQScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> lookupTable,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq,
        int M,
        int k,
        int nprobe,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf);

/// Compute ||v||² norms for each vector.  Result is written to normsBuf
/// (nb float).  Useful for caching centroid norms across searches.
bool runMetalComputeNorms(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors,
        int nb,
        int d,
        id<MTLBuffer> normsBuf);

/// Full IVF search in a single command buffer: coarse quantisation (distance
/// matrix + top-nprobe) followed by ivf_scan_list + ivf_merge_lists.  Avoids
/// the GPU sync point between coarse quant and IVF scan.
///
/// @param centroids       Centroid vectors (nlist * d float)
/// @param nlist           Number of centroids / inverted lists
/// @param coarseDistBuf   Scratch (nq * nprobe float)
/// @param coarseIdxBuf    Scratch (nq * nprobe int32)
/// @param distMatrixBuf   Scratch (nq * nlist  float)
/// @param centroidNormsBuf Pre-computed ||c||² per centroid (nlist float);
///                         if non-nil and isL2, the fused l2_with_norms kernel
///                         is used instead of l2_squared_matrix.
/// @param avgListLen      Average inverted-list length (ntotal / nlist);
///                         used to select the small-list (32-thread) scan
///                         variant when lists are short.
bool runMetalIVFFlatFullSearch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        int nq,
        int d,
        int k,
        int nprobe,
        bool isL2,
        id<MTLBuffer> centroids,
        int nlist,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> coarseDistBuf,
        id<MTLBuffer> coarseIdxBuf,
        id<MTLBuffer> distMatrixBuf,
        id<MTLBuffer> centroidNormsBuf = nil,
        int avgListLen = 256,
        id<MTLBuffer> interleavedCodes = nil,
        id<MTLBuffer> interleavedCodesOffset = nil,
        bool centroidsAreFP16 = false);

// ============================================================
//  Public brute-force k-NN on raw CPU pointers (mirrors CUDA bfKnn)
// ============================================================

class MetalResources; // forward declaration

/// Brute-force k-nearest-neighbor search on externally-provided data.
/// Vectors and queries are row-major float32 CPU pointers; results are
/// written to caller-provided CPU arrays. All GPU buffer management is
/// handled internally.
///
/// @param resources  Metal resources (device + queue)
/// @param vectors    Database vectors, row-major (numVectors x dims) float
/// @param numVectors Number of database vectors
/// @param queries    Query vectors, row-major (numQueries x dims) float
/// @param numQueries Number of query vectors
/// @param dims       Vector dimensionality
/// @param k          Number of nearest neighbors to return (must be > 0)
/// @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
/// @param outDistances  Output distances, row-major (numQueries x k) float
/// @param outIndices    Output indices, row-major (numQueries x k) idx_t
void bfKnn(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        faiss::MetricType metric,
        float* outDistances,
        idx_t* outIndices);

} // namespace gpu_metal
} // namespace faiss
