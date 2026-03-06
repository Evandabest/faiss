// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Orchestration layer for Metal distance computation and IVF search.
 * All kernel dispatch is delegated to MetalKernels; this file owns
 * command buffer lifecycle, tiling strategy, and buffer allocation.
 */

#import "MetalDistance.h"
#import "MetalKernels.h"
#include <algorithm>

namespace faiss {
namespace gpu_metal {

void chooseTileSize(
        int nq, int nb, int d,
        size_t elementSize, size_t availableMem,
        int& tileRows, int& tileCols) {
    size_t targetUsage = 512 * 1024 * 1024;
    targetUsage /= 2;
    targetUsage /= elementSize;
    int preferredTileRows = (d <= 32) ? 1024 : 512;
    tileRows = std::min(preferredTileRows, nq);
    tileCols = std::min((int)(targetUsage / preferredTileRows), nb);
    if (tileRows < 1) tileRows = 1;
    if (tileCols < 1) tileCols = 1;
}

int getMetalDistanceMaxK() {
    return MetalKernels::kMaxK;
}

// ============================================================
//  runMetalDistance
// ============================================================

bool runMetalDistance(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq, int nb, int d, int k,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices) {
    if (!device || !queue || !queries || !vectors ||
        !outDistances || !outIndices)
        return false;
    if (k <= 0 || k > MetalKernels::kMaxK) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    int tileRows, tileCols;
    chooseTileSize(nq, nb, d, sizeof(float), 512ULL * 1024 * 1024,
                   tileRows, tileCols);
    bool needsTiling = (tileCols < nb || tileRows < nq);
    int K_prime = needsTiling ? MetalKernels::computeKPrimeForTiling(k) : k;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    if (!needsTiling && k <= 1024 && d <= 2048) {
        K.encodeFusedDistTopK(enc, queries, vectors, outDistances, outIndices,
                              nq, nb, d, k, isL2);
    } else if (!needsTiling) {
        id<MTLBuffer> distMat = [device
                newBufferWithLength:(size_t)nq * nb * sizeof(float)
                            options:MTLResourceStorageModeShared];
        if (!distMat) { [enc endEncoding]; return false; }

        if (isL2) K.encodeL2SquaredMatrix(enc, queries, vectors, distMat, nq, nb, d);
        else      K.encodeIPMatrix(enc, queries, vectors, distMat, nq, nb, d);

        K.encodeTopKThreadgroup(enc, distMat, outDistances, outIndices, nq, nb, k, isL2);
    } else {
        int numRowTiles = (nq + tileRows - 1) / tileRows;
        int numColTiles = (nb + tileCols - 1) / tileCols;

        id<MTLBuffer> tileDistBuf = [device
                newBufferWithLength:(size_t)nq * numColTiles * K_prime * sizeof(float)
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> tileIdxBuf = [device
                newBufferWithLength:(size_t)nq * numColTiles * K_prime * sizeof(int32_t)
                            options:MTLResourceStorageModeShared];
        if (!tileDistBuf || !tileIdxBuf) { [enc endEncoding]; return false; }

        for (int tr = 0; tr < numRowTiles; tr++) {
            int curQS = std::min(tileRows, nq - tr * tileRows);
            size_t qBase = (size_t)(tr * tileRows) * numColTiles * K_prime;

            for (int tc = 0; tc < numColTiles; tc++) {
                int curVS = std::min(tileCols, nb - tc * tileCols);
                size_t qOff = (size_t)(tr * tileRows) * d * sizeof(float);
                size_t vOff = (size_t)(tc * tileCols) * d * sizeof(float);

                id<MTLBuffer> distTile = [device
                        newBufferWithLength:(size_t)curQS * curVS * sizeof(float)
                                    options:MTLResourceStorageModeShared];
                if (!distTile) { [enc endEncoding]; return false; }

                if (isL2) K.encodeL2SquaredMatrix(enc, queries, vectors, distTile,
                                                   curQS, curVS, d, qOff, vOff);
                else      K.encodeIPMatrix(enc, queries, vectors, distTile,
                                            curQS, curVS, d, qOff, vOff);

                id<MTLBuffer> topDist = [device
                        newBufferWithLength:(size_t)curQS * K_prime * sizeof(float)
                                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> topIdx = [device
                        newBufferWithLength:(size_t)curQS * K_prime * sizeof(int32_t)
                                    options:MTLResourceStorageModeShared];
                if (!topDist || !topIdx) { [enc endEncoding]; return false; }

                K.encodeTopKThreadgroup(enc, distTile, topDist, topIdx,
                                        curQS, curVS, K_prime, isL2);

                // Blit per-tile results into the combined buffer
                [enc endEncoding];
                id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                for (int q = 0; q < curQS; q++) {
                    size_t src = (size_t)q * K_prime;
                    size_t dst = qBase + (size_t)q * numColTiles * K_prime
                                 + (size_t)tc * K_prime;
                    [blit copyFromBuffer:topDist sourceOffset:src * sizeof(float)
                                toBuffer:tileDistBuf destinationOffset:dst * sizeof(float)
                                    size:K_prime * sizeof(float)];
                    [blit copyFromBuffer:topIdx sourceOffset:src * sizeof(int32_t)
                                toBuffer:tileIdxBuf destinationOffset:dst * sizeof(int32_t)
                                    size:K_prime * sizeof(int32_t)];
                }
                [blit endEncoding];
                enc = [cmdBuf computeCommandEncoder];
            }

            // Adjust indices to global offsets
            if (numColTiles > 1) {
                K.encodeIncrementIndex(enc, tileIdxBuf, curQS, K_prime,
                                       tileCols, numColTiles,
                                       qBase * sizeof(int32_t));
            }

            // Merge per-tile results
            if (numColTiles > 1) {
                id<MTLBuffer> mBufA_D = tileDistBuf, mBufA_I = tileIdxBuf;
                id<MTLBuffer> mBufB_D = [device
                        newBufferWithLength:(size_t)curQS * numColTiles * K_prime * sizeof(float)
                                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> mBufB_I = [device
                        newBufferWithLength:(size_t)curQS * numColTiles * K_prime * sizeof(int32_t)
                                    options:MTLResourceStorageModeShared];
                if (!mBufB_D || !mBufB_I) { [enc endEncoding]; return false; }

                int nLists = numColTiles;
                bool useA = true;
                size_t stride = (size_t)numColTiles * K_prime;

                while (nLists > 1) {
                    int nPairs = nLists / 2;
                    int nRem   = nLists % 2;

                    for (int p = 0; p < nPairs; p++) {
                        id<MTLBuffer> sD = useA ? mBufA_D : mBufB_D;
                        id<MTLBuffer> sI = useA ? mBufA_I : mBufB_I;
                        id<MTLBuffer> dD = useA ? mBufB_D : mBufA_D;
                        id<MTLBuffer> dI = useA ? mBufB_I : mBufA_I;

                        for (int q = 0; q < curQS; q++) {
                            size_t oA = (qBase + q * stride + p * 2 * K_prime) * sizeof(float);
                            size_t oB = (qBase + q * stride + (p * 2 + 1) * K_prime) * sizeof(float);
                            size_t oD = (qBase + q * stride + p * K_prime) * sizeof(float);
                            size_t oAi = (qBase + q * stride + p * 2 * K_prime) * sizeof(int32_t);
                            size_t oBi = (qBase + q * stride + (p * 2 + 1) * K_prime) * sizeof(int32_t);
                            size_t oDi = (qBase + q * stride + p * K_prime) * sizeof(int32_t);
                            K.encodeMergeTwoSorted(enc, sD, sI, sD, sI, dD, dI,
                                                    1, K_prime, isL2,
                                                    oA, oAi, oB, oBi, oD, oDi);
                        }
                    }

                    if (nRem > 0) {
                        [enc endEncoding];
                        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                        int ri = nPairs * 2;
                        id<MTLBuffer> sD = useA ? mBufA_D : mBufB_D;
                        id<MTLBuffer> sI = useA ? mBufA_I : mBufB_I;
                        id<MTLBuffer> dD = useA ? mBufB_D : mBufA_D;
                        id<MTLBuffer> dI = useA ? mBufB_I : mBufA_I;
                        for (int q = 0; q < curQS; q++) {
                            size_t so = (qBase + q * stride + ri * K_prime);
                            size_t dso = (qBase + q * stride + nPairs * K_prime);
                            [blit copyFromBuffer:sD sourceOffset:so * sizeof(float)
                                        toBuffer:dD destinationOffset:dso * sizeof(float)
                                            size:K_prime * sizeof(float)];
                            [blit copyFromBuffer:sI sourceOffset:so * sizeof(int32_t)
                                        toBuffer:dI destinationOffset:dso * sizeof(int32_t)
                                            size:K_prime * sizeof(int32_t)];
                        }
                        [blit endEncoding];
                        enc = [cmdBuf computeCommandEncoder];
                    }

                    nLists = nPairs + nRem;
                    useA = !useA;
                }

                id<MTLBuffer> fD = useA ? mBufA_D : mBufB_D;
                id<MTLBuffer> fI = useA ? mBufA_I : mBufB_I;
                if (K_prime > k) {
                    for (int q = 0; q < curQS; q++) {
                        size_t si = (qBase + q * stride) * sizeof(float);
                        size_t sii = (qBase + q * stride) * sizeof(int32_t);
                        size_t di = (size_t)(tr * tileRows + q) * k * sizeof(float);
                        size_t dii = (size_t)(tr * tileRows + q) * k * sizeof(int32_t);
                        K.encodeTrimKToK(enc, fD, fI, outDistances, outIndices,
                                         1, K_prime, k, isL2,
                                         si, sii, di, dii);
                    }
                } else {
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                    for (int q = 0; q < curQS; q++) {
                        size_t so = (qBase + q * stride);
                        size_t dso = (size_t)(tr * tileRows + q) * k;
                        [blit copyFromBuffer:fD sourceOffset:so * sizeof(float)
                                    toBuffer:outDistances destinationOffset:dso * sizeof(float)
                                        size:k * sizeof(float)];
                        [blit copyFromBuffer:fI sourceOffset:so * sizeof(int32_t)
                                    toBuffer:outIndices destinationOffset:dso * sizeof(int32_t)
                                        size:k * sizeof(int32_t)];
                    }
                    [blit endEncoding];
                    enc = [cmdBuf computeCommandEncoder];
                }
            } else {
                // Single column tile
                if (K_prime > k) {
                    size_t si = (size_t)(tr * tileRows) * K_prime * sizeof(float);
                    size_t sii = (size_t)(tr * tileRows) * K_prime * sizeof(int32_t);
                    size_t di = (size_t)(tr * tileRows) * k * sizeof(float);
                    size_t dii = (size_t)(tr * tileRows) * k * sizeof(int32_t);
                    K.encodeTrimKToK(enc, tileDistBuf, tileIdxBuf,
                                     outDistances, outIndices,
                                     curQS, K_prime, k, isL2,
                                     si, sii, di, dii);
                } else {
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                    size_t so = (size_t)(tr * tileRows) * K_prime;
                    size_t dso = (size_t)(tr * tileRows) * k;
                    [blit copyFromBuffer:tileDistBuf sourceOffset:so * sizeof(float)
                                toBuffer:outDistances destinationOffset:dso * sizeof(float)
                                    size:curQS * k * sizeof(float)];
                    [blit copyFromBuffer:tileIdxBuf sourceOffset:so * sizeof(int32_t)
                                toBuffer:outIndices destinationOffset:dso * sizeof(int32_t)
                                    size:curQS * k * sizeof(int32_t)];
                    [blit endEncoding];
                    enc = [cmdBuf computeCommandEncoder];
                }
            }
        }
    }

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return true;
}

// ============================================================
//  Convenience wrappers
// ============================================================

bool runMetalL2Distance(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> vectors,
        int nq, int nb, int d, int k,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices) {
    return runMetalDistance(device, queue, queries, vectors,
                           nq, nb, d, k, true, outDistances, outIndices);
}

bool runMetalIPDistance(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> vectors,
        int nq, int nb, int d, int k,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices) {
    return runMetalDistance(device, queue, queries, vectors,
                           nq, nb, d, k, false, outDistances, outIndices);
}

// ============================================================
//  runMetalIVFFlatScan
// ============================================================

bool runMetalIVFFlatScan(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> listOffset, id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq, int d, int k, int nprobe, bool isL2,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf, id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> interleavedCodes,
        id<MTLBuffer> interleavedCodesOffset) {
    if (!device || !queue || !queries || !codes || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    bool useIL = (interleavedCodes != nil && interleavedCodesOffset != nil);
    IVFScanVariant variant = useIL ? IVFScanVariant::Interleaved
                                   : IVFScanVariant::Standard;

    uint32_t sp[5] = {(uint32_t)nq, (uint32_t)d, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeIVFScanList(enc, variant, queries,
                        useIL ? interleavedCodes : codes,
                        ids, listOffset, listLength, coarseAssign,
                        perListDistBuf, perListIdxBuf, paramsBuf,
                        nq, nprobe,
                        useIL ? interleavedCodesOffset : nil);
    K.encodeIVFMergeLists(enc, perListDistBuf, perListIdxBuf,
                          outDistances, outIndices, paramsBuf, nq);

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

// ============================================================
//  runMetalComputeNorms
// ============================================================

bool runMetalComputeNorms(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors, int nb, int d,
        id<MTLBuffer> normsBuf) {
    if (!device || !queue || !vectors || !normsBuf || nb <= 0 || d <= 0)
        return false;
    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    K.encodeComputeNorms(enc, vectors, normsBuf, nb, d);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

// ============================================================
//  runMetalIVFFlatFullSearch
// ============================================================

bool runMetalIVFFlatFullSearch(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        int nq, int d, int k, int nprobe, bool isL2,
        id<MTLBuffer> centroids, int nlist,
        id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> listOffset, id<MTLBuffer> listLength,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf, id<MTLBuffer> perListIdxBuf,
        id<MTLBuffer> coarseDistBuf, id<MTLBuffer> coarseIdxBuf,
        id<MTLBuffer> distMatrixBuf,
        id<MTLBuffer> centroidNormsBuf,
        int avgListLen,
        id<MTLBuffer> interleavedCodes,
        id<MTLBuffer> interleavedCodesOffset) {
    if (!device || !queue || !queries || !centroids || !codes || !ids ||
        !listOffset || !listLength || !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf ||
        !coarseDistBuf || !coarseIdxBuf || !distMatrixBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0 || nlist <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    bool useIL = (interleavedCodes != nil && interleavedCodesOffset != nil);
    bool useSmall = (!useIL && avgListLen <= 64);
    IVFScanVariant scanV = useIL   ? IVFScanVariant::Interleaved
                         : useSmall ? IVFScanVariant::Small
                                    : IVFScanVariant::Standard;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // Step 1: coarse distance matrix
    bool fusedL2 = isL2 && centroidNormsBuf != nil;
    if (fusedL2)
        K.encodeL2WithNorms(enc, queries, centroids, distMatrixBuf,
                            centroidNormsBuf, nq, nlist, d);
    else if (isL2)
        K.encodeL2SquaredMatrix(enc, queries, centroids, distMatrixBuf,
                                nq, nlist, d);
    else
        K.encodeIPMatrix(enc, queries, centroids, distMatrixBuf,
                         nq, nlist, d);

    // Step 2: coarse top-nprobe (parallel threadgroup select, covers up to k=2048)
    K.encodeTopKThreadgroup(enc, distMatrixBuf, coarseDistBuf, coarseIdxBuf,
                            nq, nlist, nprobe, isL2);

    // Step 3: IVF scan
    uint32_t sp[5] = {(uint32_t)nq, (uint32_t)d, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) { [enc endEncoding]; return false; }

    K.encodeIVFScanList(enc, scanV, queries,
                        useIL ? interleavedCodes : codes,
                        ids, listOffset, listLength, coarseIdxBuf,
                        perListDistBuf, perListIdxBuf, paramsBuf,
                        nq, nprobe,
                        useIL ? interleavedCodesOffset : nil);

    // Step 4: merge
    K.encodeIVFMergeLists(enc, perListDistBuf, perListIdxBuf,
                          outDistances, outIndices, paramsBuf, nq);

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

} // namespace gpu_metal
} // namespace faiss
