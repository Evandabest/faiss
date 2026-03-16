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
#import "MetalResources.h"
#import <Foundation/Foundation.h>
#import <mach/host_info.h>
#import <mach/mach.h>
#import <mach/vm_statistics.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace faiss {
namespace gpu_metal {

namespace {

/// Default tiling budget when system query fails or returns 0 (matches previous hardcoded behavior).
constexpr size_t kDefaultTilingBudgetBytes = 256ULL * 1024 * 1024;

/// Minimum budget so we don't over-tile on very low memory.
constexpr size_t kMinTilingBudgetBytes = 64ULL * 1024 * 1024;

/// Cap so we don't assume we own all free memory (other processes need headroom).
constexpr size_t kMaxTilingBudgetBytes = 2ULL * 1024 * 1024 * 1024 * 4;

/// Returns a byte budget for distance tiling based on system available memory.
/// Uses Mach host_statistics64 (free + inactive pages), with fallback to a fraction of
/// physical memory. Result is clamped to [kMinTilingBudgetBytes, kMaxTilingBudgetBytes].
/// If FAISS_METAL_TILING_MEMORY_BYTES is set, that value is used instead (for tests/debug).
size_t getAvailableMemoryForTiling() {
    const char* envBytes = std::getenv("FAISS_METAL_TILING_MEMORY_BYTES");
    if (envBytes && envBytes[0] != '\0') {
        char* end = nullptr;
        unsigned long long val = std::strtoull(envBytes, &end, 10);
        if (end != envBytes && val != 0) {
            return static_cast<size_t>(std::min(std::max(val, (unsigned long long)kMinTilingBudgetBytes),
                                                (unsigned long long)kMaxTilingBudgetBytes));
        }
    }

    size_t availableBytes = 0;
    size_t physicalBytes = 0;

    // Mach host VM stats (free + inactive pages).
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count);
    if (kr == KERN_SUCCESS) {
        vm_size_t pageSize = 0;
        if (host_page_size(host, &pageSize) == KERN_SUCCESS && pageSize > 0) {
            uint64_t freePages = vm_stats.free_count + vm_stats.inactive_count;
            availableBytes = static_cast<size_t>(freePages * pageSize);
        }
    }

    // Physical memory for ceiling and fallback.
    uint64_t phys = [[NSProcessInfo processInfo] physicalMemory];
    physicalBytes = static_cast<size_t>(phys);

    if (availableBytes == 0 && physicalBytes > 0) {
        availableBytes = physicalBytes / 4;
    }

    if (physicalBytes > 0) {
        size_t cap = std::min(physicalBytes / 4, kMaxTilingBudgetBytes);
        availableBytes = std::min(availableBytes, cap);
    }
    availableBytes = std::min(availableBytes, kMaxTilingBudgetBytes);

    // Floor and final default.
    if (availableBytes < kMinTilingBudgetBytes) {
        availableBytes = kMinTilingBudgetBytes;
    }
    if (availableBytes == 0) {
        availableBytes = kDefaultTilingBudgetBytes;
    }
    
    //printf("Available bytes: %zu \n", availableBytes);
    return availableBytes;
}

class TempBufferArena {
   public:
    TempBufferArena(
            id<MTLDevice> device,
            std::shared_ptr<MetalResources> resources)
            : device_(device), resources_(resources) {}

    id<MTLBuffer> alloc(size_t bytes) {
        id<MTLBuffer> buf = nil;
        if (resources_) {
            buf = resources_->allocBuffer(
                    bytes, MetalAllocType::TemporaryMemoryBuffer);
            if (buf) {
                pooledBuffers_.push_back(buf);
                return buf;
            }
        }
        return [device_ newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
    }

    ~TempBufferArena() {
        if (!resources_) {
            return;
        }
        for (id<MTLBuffer> b : pooledBuffers_) {
            resources_->deallocBuffer(b, MetalAllocType::TemporaryMemoryBuffer);
        }
    }

   private:
    id<MTLDevice> device_;
    std::shared_ptr<MetalResources> resources_;
    std::vector<id<MTLBuffer>> pooledBuffers_;
};

} // namespace

void chooseTileSize(
        int nq, int nb, int d,
        size_t elementSize, size_t availableMem,
        int& tileRows, int& tileCols) {

    size_t targetUsageBytes = availableMem/ 2;
    if (targetUsageBytes < kMinTilingBudgetBytes) {
        targetUsageBytes = kMinTilingBudgetBytes;
    }
    size_t targetUsage = targetUsageBytes / elementSize;
    if (targetUsage == 0) targetUsage = 1;

    //printf("Target usage bytes %zu, Target used %zu \n", targetUsageBytes, targetUsage);

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
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources) {
    if (!device || !queue || !queries || !vectors ||
        !outDistances || !outIndices)
        return false;
    if (k <= 0 || k > MetalKernels::kMaxK) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    size_t availableMem = getAvailableMemoryForTiling();
    if (availableMem == 0) {
        availableMem = kDefaultTilingBudgetBytes;
    }
    int tileRows, tileCols;
    chooseTileSize(nq, nb, d, sizeof(float), availableMem,
                   tileRows, tileCols);
    bool needsTiling = (tileCols < nb || tileRows < nq);
    int K_prime = needsTiling ? MetalKernels::computeKPrimeForTiling(k) : k;
    TempBufferArena arena(device, resources);

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    if (!needsTiling && k <= 1024 && d <= 2048) {
        K.encodeFusedDistTopK(enc, queries, vectors, outDistances, outIndices,
                              nq, nb, d, k, isL2);
    } else if (!needsTiling) {
        id<MTLBuffer> distMat =
                arena.alloc((size_t)nq * nb * sizeof(float));
        if (!distMat) { [enc endEncoding]; return false; }

        if (isL2) K.encodeL2SquaredMatrix(enc, queries, vectors, distMat, nq, nb, d);
        else      K.encodeIPMatrix(enc, queries, vectors, distMat, nq, nb, d);

        K.encodeTopKThreadgroup(enc, distMat, outDistances, outIndices, nq, nb, k, isL2);
    } else {
        int numRowTiles = (nq + tileRows - 1) / tileRows;
        int numColTiles = (nb + tileCols - 1) / tileCols;

        id<MTLBuffer> tileDistBuf =
                arena.alloc((size_t)nq * numColTiles * K_prime * sizeof(float));
        id<MTLBuffer> tileIdxBuf =
                arena.alloc((size_t)nq * numColTiles * K_prime * sizeof(int32_t));
        if (!tileDistBuf || !tileIdxBuf) { [enc endEncoding]; return false; }

        for (int tr = 0; tr < numRowTiles; tr++) {
            int curQS = std::min(tileRows, nq - tr * tileRows);
            size_t qBase = (size_t)(tr * tileRows) * numColTiles * K_prime;

            for (int tc = 0; tc < numColTiles; tc++) {
                int curVS = std::min(tileCols, nb - tc * tileCols);
                size_t qOff = (size_t)(tr * tileRows) * d * sizeof(float);
                size_t vOff = (size_t)(tc * tileCols) * d * sizeof(float);

                id<MTLBuffer> distTile =
                        arena.alloc((size_t)curQS * curVS * sizeof(float));
                if (!distTile) { [enc endEncoding]; return false; }

                if (isL2) K.encodeL2SquaredMatrix(enc, queries, vectors, distTile,
                                                   curQS, curVS, d, qOff, vOff);
                else      K.encodeIPMatrix(enc, queries, vectors, distTile,
                                            curQS, curVS, d, qOff, vOff);

                id<MTLBuffer> topDist =
                        arena.alloc((size_t)curQS * K_prime * sizeof(float));
                id<MTLBuffer> topIdx =
                        arena.alloc((size_t)curQS * K_prime * sizeof(int32_t));
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
                id<MTLBuffer> mBufB_D = arena.alloc(
                        (size_t)curQS * numColTiles * K_prime * sizeof(float));
                id<MTLBuffer> mBufB_I = arena.alloc(
                        (size_t)curQS * numColTiles * K_prime * sizeof(int32_t));
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
//  runMetalDistanceFP16
// ============================================================

bool runMetalDistanceFP16(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        int nq, int nb, int d, int k,
        bool isL2,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources) {
    if (!device || !queue || !queries || !vectors ||
        !outDistances || !outIndices)
        return false;
    if (k <= 0 || k > MetalKernels::kMaxK) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    size_t availableMem = getAvailableMemoryForTiling();
    if (availableMem == 0) {
        availableMem = kDefaultTilingBudgetBytes;
    }
    int tileRows, tileCols;
    chooseTileSize(nq, nb, d, sizeof(uint16_t), availableMem,
                   tileRows, tileCols);
    bool needsTiling = (tileCols < nb || tileRows < nq);
    int K_prime = needsTiling ? MetalKernels::computeKPrimeForTiling(k) : k;
    TempBufferArena arena(device, resources);

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    if (!needsTiling && k <= 1024 && d <= 2048) {
        K.encodeFusedDistTopKFP16(enc, queries, vectors, outDistances, outIndices,
                                  nq, nb, d, k, isL2);
    } else if (!needsTiling) {
        id<MTLBuffer> distMat =
                arena.alloc((size_t)nq * nb * sizeof(float));
        if (!distMat) { [enc endEncoding]; return false; }

        if (isL2) K.encodeL2SquaredMatrixFP16(enc, queries, vectors, distMat, nq, nb, d);
        else      K.encodeIPMatrixFP16(enc, queries, vectors, distMat, nq, nb, d);

        K.encodeTopKThreadgroup(enc, distMat, outDistances, outIndices, nq, nb, k, isL2);
    } else {
        int numRowTiles = (nq + tileRows - 1) / tileRows;
        int numColTiles = (nb + tileCols - 1) / tileCols;

        id<MTLBuffer> tileDistBuf =
                arena.alloc((size_t)nq * numColTiles * K_prime * sizeof(float));
        id<MTLBuffer> tileIdxBuf =
                arena.alloc((size_t)nq * numColTiles * K_prime * sizeof(int32_t));
        if (!tileDistBuf || !tileIdxBuf) { [enc endEncoding]; return false; }

        for (int tr = 0; tr < numRowTiles; tr++) {
            int curQS = std::min(tileRows, nq - tr * tileRows);
            size_t qBase = (size_t)(tr * tileRows) * numColTiles * K_prime;

            for (int tc = 0; tc < numColTiles; tc++) {
                int curVS = std::min(tileCols, nb - tc * tileCols);
                size_t qOff = (size_t)(tr * tileRows) * d * sizeof(float);
                size_t vOff = (size_t)(tc * tileCols) * d * sizeof(uint16_t);

                id<MTLBuffer> distTile =
                        arena.alloc((size_t)curQS * curVS * sizeof(float));
                if (!distTile) { [enc endEncoding]; return false; }

                if (isL2) K.encodeL2SquaredMatrixFP16(enc, queries, vectors, distTile,
                                                       curQS, curVS, d, qOff, vOff);
                else      K.encodeIPMatrixFP16(enc, queries, vectors, distTile,
                                                curQS, curVS, d, qOff, vOff);

                id<MTLBuffer> topDist =
                        arena.alloc((size_t)curQS * K_prime * sizeof(float));
                id<MTLBuffer> topIdx =
                        arena.alloc((size_t)curQS * K_prime * sizeof(int32_t));
                if (!topDist || !topIdx) { [enc endEncoding]; return false; }

                K.encodeTopKThreadgroup(enc, distTile, topDist, topIdx,
                                        curQS, curVS, K_prime, isL2);

                if (numColTiles > 1) {
                    K.encodeIncrementIndex(enc, tileIdxBuf, curQS, K_prime,
                                           tileCols, numColTiles,
                                           qBase * sizeof(int32_t));
                }

                size_t dstOff = (qBase + (size_t)tc * K_prime) * sizeof(float);
                size_t dstOffIdx = (qBase + (size_t)tc * K_prime) * sizeof(int32_t);
                K.encodeMergeTwoSorted(enc,
                        tileDistBuf, tileIdxBuf,
                        topDist, topIdx,
                        tileDistBuf, tileIdxBuf,
                        curQS, K_prime, K_prime, isL2,
                        dstOff, dstOffIdx);
            }
        }

        K.encodeTrimKToK(enc, tileDistBuf, tileIdxBuf,
                          outDistances, outIndices,
                          nq, K_prime, k, isL2);
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
//  bfKnn — public brute-force k-NN on raw CPU pointers
// ============================================================

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
        idx_t* outIndices) {
    FAISS_THROW_IF_NOT(resources && resources->isAvailable());
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(queries);
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(outIndices);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(numVectors > 0);
    FAISS_THROW_IF_NOT(numQueries > 0);
    FAISS_THROW_IF_NOT(dims > 0);
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);

    id<MTLDevice> device = resources->getDevice();
    id<MTLCommandQueue> queue = resources->getCommandQueue();

    const size_t vecBytes = (size_t)numVectors * dims * sizeof(float);
    const size_t qBytes   = (size_t)numQueries * dims * sizeof(float);
    const size_t distBytes = (size_t)numQueries * k * sizeof(float);
    const size_t idxBytes  = (size_t)numQueries * k * sizeof(int32_t);

    id<MTLBuffer> vecBuf  = resources->allocBuffer(vecBytes,  MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> qBuf    = resources->allocBuffer(qBytes,    MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> distBuf = resources->allocBuffer(distBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> idxBuf  = resources->allocBuffer(idxBytes,  MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(vecBuf && qBuf && distBuf && idxBuf,
                           "bfKnn: failed to allocate Metal buffers");

    std::memcpy([vecBuf contents], vectors, vecBytes);
    std::memcpy([qBuf contents], queries, qBytes);

    bool ok = runMetalDistance(
            device, queue, qBuf, vecBuf,
            (int)numQueries, (int)numVectors, dims, k,
            (metric == METRIC_L2),
            distBuf, idxBuf, resources);

    resources->deallocBuffer(vecBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(qBuf,   MetalAllocType::TemporaryMemoryBuffer);

    if (!ok) {
        resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
        resources->deallocBuffer(idxBuf,  MetalAllocType::TemporaryMemoryBuffer);
        FAISS_THROW_MSG("bfKnn: Metal distance computation failed");
    }

    std::memcpy(outDistances, [distBuf contents], distBytes);

    const int32_t* gpuIdx = (const int32_t*)[idxBuf contents];
    for (idx_t i = 0; i < (idx_t)numQueries * k; ++i) {
        outIndices[i] = (idx_t)gpuIdx[i];
    }

    resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(idxBuf,  MetalAllocType::TemporaryMemoryBuffer);
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
//  runMetalIVFSQScan
// ============================================================

bool runMetalIVFSQScan(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> listOffset, id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq, int d, int k, int nprobe, bool isL2,
        MetalSQType sqType,
        id<MTLBuffer> sqTables,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf, id<MTLBuffer> perListIdxBuf) {
    if (!device || !queue || !queries || !codes || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0) return false;
    if (sqType == MetalSQType::SQ8 && !sqTables) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    IVFScanVariant variant = (sqType == MetalSQType::SQ8)
                                 ? IVFScanVariant::SQ8
                                 : IVFScanVariant::FP16;

    uint32_t sp[5] = {(uint32_t)nq, (uint32_t)d, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeIVFScanList(enc, variant, queries, codes,
                        ids, listOffset, listLength, coarseAssign,
                        perListDistBuf, perListIdxBuf, paramsBuf,
                        nq, nprobe,
                        nil /* ilCodesOffset */,
                        sqTables);
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
//  runMetalIVFPQScan
// ============================================================

bool runMetalIVFPQScan(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> lookupTable,
        id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> listOffset, id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        int nq, int M, int k, int nprobe, bool isL2,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf, id<MTLBuffer> perListIdxBuf) {
    if (!device || !queue || !lookupTable || !codes || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0 || M <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    // params: [nq, M, k, nprobe, want_min]
    uint32_t sp[5] = {(uint32_t)nq, (uint32_t)M, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeIVFPQScanList(enc, lookupTable, codes, ids,
                           listOffset, listLength, coarseAssign,
                           perListDistBuf, perListIdxBuf, paramsBuf,
                           nq, nprobe);

    // Reuse the IVF merge kernel. It expects params: [nq, d, k, nprobe, want_min].
    // We overwrite params[1] (M → d is unused by merge, but nq/k/nprobe/want_min match).
    K.encodeIVFMergeLists(enc, perListDistBuf, perListIdxBuf,
                          outDistances, outIndices, paramsBuf, nq);

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

// ============================================================
//  runMetalHammingDistance
// ============================================================

bool runMetalHammingDistance(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> database,
        int nq, int nb, int code_size, int k,
        id<MTLBuffer> outDist, id<MTLBuffer> outIdx) {
    if (!device || !queue || !queries || !database ||
        !outDist || !outIdx)
        return false;
    if (nq <= 0 || nb <= 0 || code_size <= 0 || k <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    uint32_t sp[4] = {(uint32_t)nq, (uint32_t)nb,
                      (uint32_t)code_size, (uint32_t)k};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeHammingDistanceTopK(enc, queries, database,
                                 outDist, outIdx, paramsBuf, nq);

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
        id<MTLBuffer> interleavedCodesOffset,
        bool centroidsAreFP16) {
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
    if (centroidsAreFP16) {
        if (isL2)
            K.encodeL2SquaredMatrixFP16(enc, queries, centroids, distMatrixBuf,
                                         nq, nlist, d);
        else
            K.encodeIPMatrixFP16(enc, queries, centroids, distMatrixBuf,
                                  nq, nlist, d);
    } else {
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
    }

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
