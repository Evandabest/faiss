// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MetalKernels: typed wrapper around all Metal compute kernels.
 * Owns library compilation, pipeline caching, buffer binding, and dispatch.
 * Call sites use encode*() methods instead of raw pipeline/buffer/dispatch.
 */

#pragma once

#import <Metal/Metal.h>
#include <string>
#include <unordered_map>

namespace faiss {
namespace gpu_metal {

enum class IVFScanVariant { Standard, Small, Interleaved, SQ8, FP16 };

class MetalKernels {
public:
    explicit MetalKernels(id<MTLDevice> device);
    ~MetalKernels();

    bool isValid() const;
    static constexpr int kMaxK = 2048;

    // ---- Distance matrix ops ----
    // Byte offsets default to 0; pass non-zero for tiled paths.

    void encodeL2SquaredMatrix(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    void encodeIPMatrix(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    void encodeL2WithNorms(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            id<MTLBuffer> vecNorms,
            int nq,
            int nb,
            int d);

    void encodeComputeNorms(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> vectors,
            id<MTLBuffer> norms,
            int nb,
            int d);

    // ---- Float16 vector distance matrix ops ----
    // Same as above but vectors buffer contains half-precision data.

    void encodeL2SquaredMatrixFP16(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    void encodeIPMatrixFP16(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    // ---- Fused distance + top-k (single-pass, no intermediate matrix, k ≤ 1024) ----

    void encodeFusedDistTopK(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int nb,
            int d,
            int k,
            bool isL2,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    void encodeFusedDistTopKFP16(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int nb,
            int d,
            int k,
            bool isL2,
            size_t queryByteOff = 0,
            size_t vectorByteOff = 0);

    // ---- Top-k ops (variant auto-selected from k, covers k ≤ 2048) ----

    void encodeTopKThreadgroup(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> distances,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int nb,
            int k,
            bool wantMin);

    // ---- Merge ops ----

    void encodeMergeTwoSorted(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> inA,
            id<MTLBuffer> inAIdx,
            id<MTLBuffer> inB,
            id<MTLBuffer> inBIdx,
            id<MTLBuffer> out,
            id<MTLBuffer> outIdx,
            int nq,
            int kActual,
            bool wantMin,
            size_t inAOff = 0,
            size_t inAIdxOff = 0,
            size_t inBOff = 0,
            size_t inBIdxOff = 0,
            size_t outOff = 0,
            size_t outIdxOff = 0);

    void encodeIncrementIndex(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> indices,
            int nq,
            int k,
            int tileCols,
            int numTiles,
            size_t indicesOff = 0);

    void encodeTrimKToK(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> inK,
            id<MTLBuffer> inIdx,
            id<MTLBuffer> outK,
            id<MTLBuffer> outIdx,
            int nq,
            int kPrime,
            int k,
            bool wantMin,
            size_t inKOff = 0,
            size_t inIdxOff = 0,
            size_t outKOff = 0,
            size_t outIdxOff = 0);

    // ---- IVF ops ----

    void encodeIVFScanList(
            id<MTLComputeCommandEncoder> enc,
            IVFScanVariant variant,
            id<MTLBuffer> queries,
            id<MTLBuffer> codes,
            id<MTLBuffer> ids,
            id<MTLBuffer> listOffset,
            id<MTLBuffer> listLength,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> paramsBuf,
            int nq,
            int nprobe,
            id<MTLBuffer> ilCodesOffset = nil,
            id<MTLBuffer> sqTables = nil);

    void encodeIVFPQScanList(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> lookupTable,
            id<MTLBuffer> codes,
            id<MTLBuffer> ids,
            id<MTLBuffer> listOffset,
            id<MTLBuffer> listLength,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> paramsBuf,
            int nq,
            int nprobe);

    void encodeIVFMergeLists(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            id<MTLBuffer> paramsBuf,
            int nq);

    // ---- Variant selection helpers (public for orchestrator) ----

    static int selectTopKVariantIndex(int k);
    static int computeKPrimeForTiling(int k);

private:
    id<MTLComputePipelineState> pipeline(const char* name);

    id<MTLDevice> device_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache_;

    static constexpr int kTopKVariantSizes[] = {
            32, 64, 128, 256, 512, 1024, 2048};
    static constexpr int kNumTopKVariants = 7;
};

MetalKernels& getMetalKernels(id<MTLDevice> device);

} // namespace gpu_metal
} // namespace faiss
