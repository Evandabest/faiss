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
#include <cstdint>
#include <memory>

namespace faiss {
namespace gpu_metal {

class MetalResources;

/// Scalar type of distance input vectors for params-based API.
enum class MetalDistanceDataType {
    F32 = 1,
    F16,
    BF16,
};

/// Scalar type of output indices for params-based API.
enum class MetalIndicesDataType {
    I64 = 1,
    I32,
};

/// Arguments to brute-force Metal k-nearest-neighbor searching.
/// This is a Metal-side counterpart to CUDA's GpuDistanceParams with a reduced
/// supported surface (L2/IP only; non-F32 inputs and column-major layout may be
/// materialized on CPU before dispatch).
struct MetalDistanceParams {
    faiss::MetricType metric = METRIC_L2;
    float metricArg = 0.0f;
    int k = 0;
    int dims = 0;

    const void* vectors = nullptr;
    MetalDistanceDataType vectorType = MetalDistanceDataType::F32;
    bool vectorsRowMajor = true;
    idx_t numVectors = 0;
    const float* vectorNorms = nullptr;

    const void* queries = nullptr;
    MetalDistanceDataType queryType = MetalDistanceDataType::F32;
    bool queriesRowMajor = true;
    idx_t numQueries = 0;

    float* outDistances = nullptr;
    bool ignoreOutDistances = false;
    MetalIndicesDataType outIndicesType = MetalIndicesDataType::I64;
    void* outIndices = nullptr;

    int device = -1;
    bool use_cuvs = false;
};

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
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources = nullptr);

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
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources = nullptr);

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
        id<MTLBuffer> interleavedCodesOffset = nil,
        bool waitForCompletion = true);

/// SQ quantizer type for IVF SQ scan.
enum class MetalSQType { SQ4, SQ6, SQ8, SQ8_DIRECT, FP16 };

/// IVF Scalar Quantizer scan: scans SQ-encoded inverted lists on the GPU.
///
/// @param sqType       Quantizer type (SQ4/SQ6/SQ8/SQ8_DIRECT/FP16)
/// @param sqTables     SQ decode tables buffer (SQ4/SQ6/SQ8:
///                     vmin[d] + vdiff[d] = 2*d floats; nil for FP16/SQ8_DIRECT)
/// @param centroids    Coarse centroids (nlist * d, float) for residual decode.
/// @param byResidual   If true, decode `code + centroid[list_no]`.
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
        id<MTLBuffer> centroids,
        bool byResidual,
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
        int avgListLen,
        bool lookupFp16,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf);

/// Build IVFPQ lookup tables on GPU:
/// outLookup layout is (nq * nprobe * M * 256) float.
bool runMetalBuildIVFPQLookupTables(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> coarseCentroids,
        id<MTLBuffer> pqCentroids,
        int nq,
        int d,
        int M,
        int nprobe,
        bool isL2,
        bool lookupFp16,
        id<MTLBuffer> outLookup);

/// Fused IVFPQ path: build LUT on GPU + scan lists + merge in one command
/// buffer to reduce synchronization overhead.
bool runMetalIVFPQFullSearch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> coarseCentroids,
        id<MTLBuffer> pqCentroids,
        id<MTLBuffer> lookupTable,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        int nq,
        int d,
        int M,
        int k,
        int nprobe,
        int avgListLen,
        bool lookupFp16,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf,
        id<MTLBuffer> perListIdxBuf);

bool runMetalConvertF32ToF16(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> srcF32,
        id<MTLBuffer> dstF16,
        size_t numElems);

/// Binary (Hamming) brute-force top-k search.
///
/// @param queries   Binary query vectors (nq * code_size bytes)
/// @param database  Binary database vectors (nb * code_size bytes)
/// @param code_size Bytes per vector (d / 8)
/// @param outDist   Output Hamming distances (nq * k int32)
/// @param outIdx    Output indices (nq * k int64)
bool runMetalHammingDistance(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> database,
        int nq,
        int nb,
        int code_size,
        int k,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx);

/// Compute ||v||² norms for each vector.  Result is written to normsBuf
/// (nb float).  Useful for caching centroid norms across searches.
bool runMetalComputeNorms(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors,
        int nb,
        int d,
        id<MTLBuffer> normsBuf,
        bool waitForCompletion = true);

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
        bool centroidsAreFP16 = false,
        bool waitForCompletion = true);

// ============================================================
//  Public brute-force k-NN on raw CPU pointers (mirrors CUDA bfKnn)
// ============================================================

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

/// Params-based bfKnn overload (GpuDistanceParams-like API).
void bfKnn(
        std::shared_ptr<MetalResources> resources,
        const MetalDistanceParams& args);

/// Memory-budgeted brute-force k-NN (Metal equivalent of CUDA bfKnn_tiling).
/// If limits are non-zero, vectors and/or queries are processed in CPU-side
/// shards and merged into final top-k results.
///
/// @param vectorsMemoryLimit Bytes budget for database vectors per shard.
///                           0 means "no explicit vector sharding".
/// @param queriesMemoryLimit Bytes budget for query+output working set per shard.
///                           0 means "no explicit query sharding".
void bfKnn_tiling(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        faiss::MetricType metric,
        float* outDistances,
        idx_t* outIndices,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit);

/// Params-based bfKnn_tiling overload (GpuDistanceParams-like API).
void bfKnn_tiling(
        std::shared_ptr<MetalResources> resources,
        const MetalDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit);

} // namespace gpu_metal
} // namespace faiss
