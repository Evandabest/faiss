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
#include <faiss/utils/bf16.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/fp16.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
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

inline bool betterDistance(float a, float b, bool isL2) {
    return isL2 ? (a < b) : (a > b);
}

void mergeShardTopK_(
        int nq,
        int k,
        bool isL2,
        const float* shardDist,
        const idx_t* shardIdx,
        idx_t indexBase,
        std::vector<float>& bestDist,
        std::vector<idx_t>& bestIdx) {
    for (int q = 0; q < nq; ++q) {
        float* rowDist = bestDist.data() + (size_t)q * k;
        idx_t* rowIdx = bestIdx.data() + (size_t)q * k;
        for (int j = 0; j < k; ++j) {
            idx_t candIdx = shardIdx[(size_t)q * k + j];
            if (candIdx < 0) {
                continue;
            }
            const float candDist = shardDist[(size_t)q * k + j];
            candIdx += indexBase;

            if (!betterDistance(candDist, rowDist[k - 1], isL2)) {
                continue;
            }

            int pos = k - 1;
            while (pos > 0 && betterDistance(candDist, rowDist[pos - 1], isL2)) {
                rowDist[pos] = rowDist[pos - 1];
                rowIdx[pos] = rowIdx[pos - 1];
                --pos;
            }
            rowDist[pos] = candDist;
            rowIdx[pos] = candIdx;
        }
    }
}

bool runMetalL2DistanceWithNorms_(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> vectorNorms,
        int nq,
        int nb,
        int d,
        int k,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        std::shared_ptr<MetalResources> resources) {
    if (!device || !queue || !queries || !vectors || !vectorNorms || !outDistances ||
        !outIndices) {
        return false;
    }
    if (k <= 0 || k > MetalKernels::kMaxK) {
        return false;
    }

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) {
        return false;
    }

    size_t availableMem = getAvailableMemoryForTiling();
    if (availableMem == 0) {
        availableMem = kDefaultTilingBudgetBytes;
    }
    int tileRows = 0;
    int tileCols = 0;
    chooseTileSize(nq, nb, d, sizeof(float), availableMem, tileRows, tileCols);

    // l2_with_norms currently has no byte-offset variant; use fast-path when
    // full matrix fits current tile budget, otherwise let caller fall back.
    const bool needsTiling = (tileCols < nb || tileRows < nq);
    if (needsTiling) {
        return false;
    }

    TempBufferArena arena(device, resources);
    id<MTLBuffer> distMat = arena.alloc((size_t)nq * nb * sizeof(float));
    if (!distMat) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    K.encodeL2WithNorms(enc, queries, vectors, distMat, vectorNorms, nq, nb, d);
    K.encodeTopKThreadgroup(enc, distMat, outDistances, outIndices, nq, nb, k, true);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

void bfKnnWithVectorNormsF32_(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        const float* vectorNorms,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        float* outDistances,
        idx_t* outIndices) {
    FAISS_THROW_IF_NOT(resources && resources->isAvailable());
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(vectorNorms);
    FAISS_THROW_IF_NOT(queries);
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(outIndices);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(numVectors > 0);
    FAISS_THROW_IF_NOT(numQueries > 0);
    FAISS_THROW_IF_NOT(dims > 0);

    id<MTLDevice> device = resources->getDevice();
    id<MTLCommandQueue> queue = resources->getCommandQueue();

    const size_t vecBytes = (size_t)numVectors * dims * sizeof(float);
    const size_t qBytes = (size_t)numQueries * dims * sizeof(float);
    const size_t normsBytes = (size_t)numVectors * sizeof(float);
    const size_t distBytes = (size_t)numQueries * k * sizeof(float);
    const size_t idxBytes = (size_t)numQueries * k * sizeof(int32_t);

    id<MTLBuffer> vecBuf = resources->allocBuffer(
            vecBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> qBuf = resources->allocBuffer(
            qBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> normsBuf = resources->allocBuffer(
            normsBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> distBuf = resources->allocBuffer(
            distBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> idxBuf = resources->allocBuffer(
            idxBytes, MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(
            vecBuf && qBuf && normsBuf && distBuf && idxBuf,
            "bfKnn(vectorNorms): failed to allocate Metal buffers");

    std::memcpy([vecBuf contents], vectors, vecBytes);
    std::memcpy([qBuf contents], queries, qBytes);
    std::memcpy([normsBuf contents], vectorNorms, normsBytes);

    bool ok = runMetalL2DistanceWithNorms_(
            device,
            queue,
            qBuf,
            vecBuf,
            normsBuf,
            (int)numQueries,
            (int)numVectors,
            dims,
            k,
            distBuf,
            idxBuf,
            resources);

    resources->deallocBuffer(vecBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(qBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(normsBuf, MetalAllocType::TemporaryMemoryBuffer);

    if (!ok) {
        // Fall back to regular L2 path if norms kernel path cannot run for this
        // shape (e.g., requires tiling).
        resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
        resources->deallocBuffer(idxBuf, MetalAllocType::TemporaryMemoryBuffer);
        bfKnn(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                METRIC_L2,
                outDistances,
                outIndices);
        return;
    }

    std::memcpy(outDistances, [distBuf contents], distBytes);

    const int32_t* gpuIdx = (const int32_t*)[idxBuf contents];
    for (idx_t i = 0; i < (idx_t)numQueries * k; ++i) {
        outIndices[i] = (idx_t)gpuIdx[i];
    }

    resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(idxBuf, MetalAllocType::TemporaryMemoryBuffer);
}

void bfKnnFP16Vectors_(
        std::shared_ptr<MetalResources> resources,
        const uint16_t* vectors,
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

    const size_t vecBytes = (size_t)numVectors * dims * sizeof(uint16_t);
    const size_t qBytes = (size_t)numQueries * dims * sizeof(float);
    const size_t distBytes = (size_t)numQueries * k * sizeof(float);
    const size_t idxBytes = (size_t)numQueries * k * sizeof(int32_t);

    id<MTLBuffer> vecBuf = resources->allocBuffer(
            vecBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> qBuf = resources->allocBuffer(
            qBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> distBuf = resources->allocBuffer(
            distBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> idxBuf = resources->allocBuffer(
            idxBytes, MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(
            vecBuf && qBuf && distBuf && idxBuf,
            "bfKnn(fp16 vectors): failed to allocate Metal buffers");

    std::memcpy([vecBuf contents], vectors, vecBytes);
    std::memcpy([qBuf contents], queries, qBytes);

    bool ok = runMetalDistanceFP16(
            device,
            queue,
            qBuf,
            vecBuf,
            (int)numQueries,
            (int)numVectors,
            dims,
            k,
            (metric == METRIC_L2),
            distBuf,
            idxBuf,
            resources);

    resources->deallocBuffer(vecBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(qBuf, MetalAllocType::TemporaryMemoryBuffer);

    if (!ok) {
        resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
        resources->deallocBuffer(idxBuf, MetalAllocType::TemporaryMemoryBuffer);
        FAISS_THROW_MSG("bfKnn(fp16 vectors): Metal distance computation failed");
    }

    std::memcpy(outDistances, [distBuf contents], distBytes);

    const int32_t* gpuIdx = (const int32_t*)[idxBuf contents];
    for (idx_t i = 0; i < (idx_t)numQueries * k; ++i) {
        outIndices[i] = (idx_t)gpuIdx[i];
    }

    resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(idxBuf, MetalAllocType::TemporaryMemoryBuffer);
}

void bfKnnSingleQueryShard_(
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
        size_t vectorsMemoryLimit) {
    if (vectorsMemoryLimit == 0) {
        bfKnn(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices);
        return;
    }

    const size_t bytesPerVector = (size_t)dims * sizeof(float);
    const size_t shardVectors = vectorsMemoryLimit / bytesPerVector;
    FAISS_THROW_IF_NOT_MSG(
            shardVectors > 0,
            "bfKnn_tiling: vectorsMemoryLimit is too low");

    if ((size_t)numVectors <= shardVectors) {
        bfKnn(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices);
        return;
    }

    const bool isL2 = (metric == METRIC_L2);
    const float sentinelDist = isL2 ? std::numeric_limits<float>::infinity()
                                    : -std::numeric_limits<float>::infinity();

    std::vector<float> bestDist((size_t)numQueries * k, sentinelDist);
    std::vector<idx_t> bestIdx((size_t)numQueries * k, -1);
    std::vector<float> shardDist((size_t)numQueries * k);
    std::vector<idx_t> shardIdx((size_t)numQueries * k);

    for (idx_t base = 0; base < numVectors; base += (idx_t)shardVectors) {
        const idx_t nThis = std::min((idx_t)shardVectors, numVectors - base);
        const float* vecPtr = vectors + (size_t)base * dims;

        bfKnn(
                resources,
                vecPtr,
                nThis,
                queries,
                numQueries,
                dims,
                k,
                metric,
                shardDist.data(),
                shardIdx.data());

        mergeShardTopK_(
                (int)numQueries,
                k,
                isL2,
                shardDist.data(),
                shardIdx.data(),
                base,
                bestDist,
                bestIdx);
    }

    std::memcpy(
            outDistances,
            bestDist.data(),
            (size_t)numQueries * k * sizeof(float));
    std::memcpy(
            outIndices,
            bestIdx.data(),
            (size_t)numQueries * k * sizeof(idx_t));
}

void bfKnnSingleQueryShardWithNorms_(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        const float* vectorNorms,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        float* outDistances,
        idx_t* outIndices,
        size_t vectorsMemoryLimit) {
    if (vectorsMemoryLimit == 0) {
        bfKnnWithVectorNormsF32_(
                resources,
                vectors,
                vectorNorms,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                outDistances,
                outIndices);
        return;
    }

    const size_t bytesPerVector = (size_t)dims * sizeof(float);
    const size_t shardVectors = vectorsMemoryLimit / bytesPerVector;
    FAISS_THROW_IF_NOT_MSG(
            shardVectors > 0,
            "bfKnn_tiling(vectorNorms): vectorsMemoryLimit is too low");

    if ((size_t)numVectors <= shardVectors) {
        bfKnnWithVectorNormsF32_(
                resources,
                vectors,
                vectorNorms,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                outDistances,
                outIndices);
        return;
    }

    std::vector<float> bestDist((size_t)numQueries * k, std::numeric_limits<float>::infinity());
    std::vector<idx_t> bestIdx((size_t)numQueries * k, -1);
    std::vector<float> shardDist((size_t)numQueries * k);
    std::vector<idx_t> shardIdx((size_t)numQueries * k);

    for (idx_t base = 0; base < numVectors; base += (idx_t)shardVectors) {
        const idx_t nThis = std::min((idx_t)shardVectors, numVectors - base);
        const float* vecPtr = vectors + (size_t)base * dims;
        const float* normPtr = vectorNorms + (size_t)base;

        bfKnnWithVectorNormsF32_(
                resources,
                vecPtr,
                normPtr,
                nThis,
                queries,
                numQueries,
                dims,
                k,
                shardDist.data(),
                shardIdx.data());

        mergeShardTopK_(
                (int)numQueries,
                k,
                true,
                shardDist.data(),
                shardIdx.data(),
                base,
                bestDist,
                bestIdx);
    }

    std::memcpy(
            outDistances,
            bestDist.data(),
            (size_t)numQueries * k * sizeof(float));
    std::memcpy(
            outIndices,
            bestIdx.data(),
            (size_t)numQueries * k * sizeof(idx_t));
}

void bfKnnSingleQueryShardFP16_(
        std::shared_ptr<MetalResources> resources,
        const uint16_t* vectors,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        faiss::MetricType metric,
        float* outDistances,
        idx_t* outIndices,
        size_t vectorsMemoryLimit) {
    if (vectorsMemoryLimit == 0) {
        bfKnnFP16Vectors_(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices);
        return;
    }

    const size_t bytesPerVector = (size_t)dims * sizeof(uint16_t);
    const size_t shardVectors = vectorsMemoryLimit / bytesPerVector;
    FAISS_THROW_IF_NOT_MSG(
            shardVectors > 0,
            "bfKnn_tiling(fp16 vectors): vectorsMemoryLimit is too low");

    if ((size_t)numVectors <= shardVectors) {
        bfKnnFP16Vectors_(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices);
        return;
    }

    const bool isL2 = (metric == METRIC_L2);
    const float sentinelDist = isL2 ? std::numeric_limits<float>::infinity()
                                    : -std::numeric_limits<float>::infinity();

    std::vector<float> bestDist((size_t)numQueries * k, sentinelDist);
    std::vector<idx_t> bestIdx((size_t)numQueries * k, -1);
    std::vector<float> shardDist((size_t)numQueries * k);
    std::vector<idx_t> shardIdx((size_t)numQueries * k);

    for (idx_t base = 0; base < numVectors; base += (idx_t)shardVectors) {
        const idx_t nThis = std::min((idx_t)shardVectors, numVectors - base);
        const uint16_t* vecPtr = vectors + (size_t)base * dims;

        bfKnnFP16Vectors_(
                resources,
                vecPtr,
                nThis,
                queries,
                numQueries,
                dims,
                k,
                metric,
                shardDist.data(),
                shardIdx.data());

        mergeShardTopK_(
                (int)numQueries,
                k,
                isL2,
                shardDist.data(),
                shardIdx.data(),
                base,
                bestDist,
                bestIdx);
    }

    std::memcpy(
            outDistances,
            bestDist.data(),
            (size_t)numQueries * k * sizeof(float));
    std::memcpy(
            outIndices,
            bestIdx.data(),
            (size_t)numQueries * k * sizeof(idx_t));
}

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
        size_t queriesMemoryLimit) {
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

    if (queriesMemoryLimit == 0) {
        bfKnnSingleQueryShard_(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices,
                vectorsMemoryLimit);
        return;
    }

    const size_t bytesPerQuery = (size_t)dims * sizeof(float) +
            (size_t)k * (sizeof(float) + sizeof(idx_t));
    const size_t shardQueries = queriesMemoryLimit / bytesPerQuery;
    FAISS_THROW_IF_NOT_MSG(
            shardQueries > 0,
            "bfKnn_tiling: queriesMemoryLimit is too low");

    for (idx_t qBase = 0; qBase < numQueries; qBase += (idx_t)shardQueries) {
        const idx_t nThisQ = std::min((idx_t)shardQueries, numQueries - qBase);
        const float* qPtr = queries + (size_t)qBase * dims;
        float* outDistPtr = outDistances + (size_t)qBase * k;
        idx_t* outIdxPtr = outIndices + (size_t)qBase * k;

        bfKnnSingleQueryShard_(
                resources,
                vectors,
                numVectors,
                qPtr,
                nThisQ,
                dims,
                k,
                metric,
                outDistPtr,
                outIdxPtr,
            vectorsMemoryLimit);
    }
}

void bfKnn_tilingFP16_(
        std::shared_ptr<MetalResources> resources,
        const uint16_t* vectors,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        faiss::MetricType metric,
        float* outDistances,
        idx_t* outIndices,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
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

    if (queriesMemoryLimit == 0) {
        bfKnnSingleQueryShardFP16_(
                resources,
                vectors,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                metric,
                outDistances,
                outIndices,
                vectorsMemoryLimit);
        return;
    }

    const size_t bytesPerQuery = (size_t)dims * sizeof(float) +
            (size_t)k * (sizeof(float) + sizeof(idx_t));
    const size_t shardQueries = queriesMemoryLimit / bytesPerQuery;
    FAISS_THROW_IF_NOT_MSG(
            shardQueries > 0,
            "bfKnn_tiling(fp16 vectors): queriesMemoryLimit is too low");

    for (idx_t qBase = 0; qBase < numQueries; qBase += (idx_t)shardQueries) {
        const idx_t nThisQ = std::min((idx_t)shardQueries, numQueries - qBase);
        const float* qPtr = queries + (size_t)qBase * dims;
        float* outDistPtr = outDistances + (size_t)qBase * k;
        idx_t* outIdxPtr = outIndices + (size_t)qBase * k;

        bfKnnSingleQueryShardFP16_(
                resources,
                vectors,
                numVectors,
                qPtr,
                nThisQ,
                dims,
                k,
                metric,
                outDistPtr,
                outIdxPtr,
                vectorsMemoryLimit);
    }
}

void bfKnn_tilingWithNormsF32_(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        const float* vectorNorms,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        int k,
        float* outDistances,
        idx_t* outIndices,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
    FAISS_THROW_IF_NOT(resources && resources->isAvailable());
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(vectorNorms);
    FAISS_THROW_IF_NOT(queries);
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(outIndices);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(numVectors > 0);
    FAISS_THROW_IF_NOT(numQueries > 0);
    FAISS_THROW_IF_NOT(dims > 0);

    if (queriesMemoryLimit == 0) {
        bfKnnSingleQueryShardWithNorms_(
                resources,
                vectors,
                vectorNorms,
                numVectors,
                queries,
                numQueries,
                dims,
                k,
                outDistances,
                outIndices,
                vectorsMemoryLimit);
        return;
    }

    const size_t bytesPerQuery = (size_t)dims * sizeof(float) +
            (size_t)k * (sizeof(float) + sizeof(idx_t));
    const size_t shardQueries = queriesMemoryLimit / bytesPerQuery;
    FAISS_THROW_IF_NOT_MSG(
            shardQueries > 0,
            "bfKnn_tiling(vectorNorms): queriesMemoryLimit is too low");

    for (idx_t qBase = 0; qBase < numQueries; qBase += (idx_t)shardQueries) {
        const idx_t nThisQ = std::min((idx_t)shardQueries, numQueries - qBase);
        const float* qPtr = queries + (size_t)qBase * dims;
        float* outDistPtr = outDistances + (size_t)qBase * k;
        idx_t* outIdxPtr = outIndices + (size_t)qBase * k;

        bfKnnSingleQueryShardWithNorms_(
                resources,
                vectors,
                vectorNorms,
                numVectors,
                qPtr,
                nThisQ,
                dims,
                k,
                outDistPtr,
                outIdxPtr,
                vectorsMemoryLimit);
    }
}

void allPairsBlockF32_(
        std::shared_ptr<MetalResources> resources,
        const float* vectors,
        const float* vectorNorms,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        faiss::MetricType metric,
        float* outDistances) {
    FAISS_THROW_IF_NOT(resources && resources->isAvailable());
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(queries);
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(numVectors > 0);
    FAISS_THROW_IF_NOT(numQueries > 0);
    FAISS_THROW_IF_NOT(dims > 0);

    id<MTLDevice> device = resources->getDevice();
    id<MTLCommandQueue> queue = resources->getCommandQueue();
    MetalKernels& K = getMetalKernels(device);
    FAISS_THROW_IF_NOT_MSG(K.isValid(), "allPairs(F32): Metal kernels unavailable");

    const size_t vecBytes = (size_t)numVectors * dims * sizeof(float);
    const size_t qBytes = (size_t)numQueries * dims * sizeof(float);
    const size_t distBytes = (size_t)numQueries * numVectors * sizeof(float);
    const size_t normsBytes = vectorNorms ? (size_t)numVectors * sizeof(float) : 0;

    id<MTLBuffer> vecBuf = resources->allocBuffer(
            vecBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> qBuf = resources->allocBuffer(
            qBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> distBuf = resources->allocBuffer(
            distBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> normsBuf = nil;
    if (vectorNorms) {
        normsBuf = resources->allocBuffer(
                normsBytes, MetalAllocType::TemporaryMemoryBuffer);
    }
    FAISS_THROW_IF_NOT_MSG(
            vecBuf && qBuf && distBuf && (!vectorNorms || normsBuf),
            "allPairs(F32): failed to allocate Metal buffers");

    std::memcpy([vecBuf contents], vectors, vecBytes);
    std::memcpy([qBuf contents], queries, qBytes);
    if (vectorNorms) {
        std::memcpy([normsBuf contents], vectorNorms, normsBytes);
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    if (metric == METRIC_L2) {
        if (vectorNorms) {
            K.encodeL2WithNorms(enc, qBuf, vecBuf, distBuf, normsBuf, (int)numQueries, (int)numVectors, dims);
        } else {
            K.encodeL2SquaredMatrix(enc, qBuf, vecBuf, distBuf, (int)numQueries, (int)numVectors, dims);
        }
    } else {
        K.encodeIPMatrix(enc, qBuf, vecBuf, distBuf, (int)numQueries, (int)numVectors, dims);
    }
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    FAISS_THROW_IF_NOT_MSG(
            cmdBuf.status == MTLCommandBufferStatusCompleted,
            "allPairs(F32): Metal distance computation failed");

    std::memcpy(outDistances, [distBuf contents], distBytes);

    resources->deallocBuffer(vecBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(qBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
    if (normsBuf) {
        resources->deallocBuffer(normsBuf, MetalAllocType::TemporaryMemoryBuffer);
    }
}

void allPairsBlockFP16Vectors_(
        std::shared_ptr<MetalResources> resources,
        const uint16_t* vectors,
        idx_t numVectors,
        const float* queries,
        idx_t numQueries,
        int dims,
        faiss::MetricType metric,
        float* outDistances) {
    FAISS_THROW_IF_NOT(resources && resources->isAvailable());
    FAISS_THROW_IF_NOT(vectors);
    FAISS_THROW_IF_NOT(queries);
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(numVectors > 0);
    FAISS_THROW_IF_NOT(numQueries > 0);
    FAISS_THROW_IF_NOT(dims > 0);

    id<MTLDevice> device = resources->getDevice();
    id<MTLCommandQueue> queue = resources->getCommandQueue();
    MetalKernels& K = getMetalKernels(device);
    FAISS_THROW_IF_NOT_MSG(K.isValid(), "allPairs(F16 vectors): Metal kernels unavailable");

    const size_t vecBytes = (size_t)numVectors * dims * sizeof(uint16_t);
    const size_t qBytes = (size_t)numQueries * dims * sizeof(float);
    const size_t distBytes = (size_t)numQueries * numVectors * sizeof(float);

    id<MTLBuffer> vecBuf = resources->allocBuffer(
            vecBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> qBuf = resources->allocBuffer(
            qBytes, MetalAllocType::TemporaryMemoryBuffer);
    id<MTLBuffer> distBuf = resources->allocBuffer(
            distBytes, MetalAllocType::TemporaryMemoryBuffer);
    FAISS_THROW_IF_NOT_MSG(
            vecBuf && qBuf && distBuf,
            "allPairs(F16 vectors): failed to allocate Metal buffers");

    std::memcpy([vecBuf contents], vectors, vecBytes);
    std::memcpy([qBuf contents], queries, qBytes);

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    if (metric == METRIC_L2) {
        K.encodeL2SquaredMatrixFP16(enc, qBuf, vecBuf, distBuf, (int)numQueries, (int)numVectors, dims);
    } else {
        K.encodeIPMatrixFP16(enc, qBuf, vecBuf, distBuf, (int)numQueries, (int)numVectors, dims);
    }
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    FAISS_THROW_IF_NOT_MSG(
            cmdBuf.status == MTLCommandBufferStatusCompleted,
            "allPairs(F16 vectors): Metal distance computation failed");

    std::memcpy(outDistances, [distBuf contents], distBytes);

    resources->deallocBuffer(vecBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(qBuf, MetalAllocType::TemporaryMemoryBuffer);
    resources->deallocBuffer(distBuf, MetalAllocType::TemporaryMemoryBuffer);
}

void allPairsDistanceTiledParams_(
        std::shared_ptr<MetalResources> resources,
        const MetalDistanceParams& args,
        const float* vectorsF32,
        const uint16_t* vectorsF16,
        const float* queriesF32,
        const float* vectorNorms,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit,
        float* outDistances) {
    FAISS_THROW_IF_NOT(outDistances);
    FAISS_THROW_IF_NOT(args.k == -1);

    const size_t vecElemBytes = (args.vectorType == MetalDistanceDataType::F16)
            ? sizeof(uint16_t)
            : sizeof(float);
    const size_t vecBytesPer = (size_t)args.dims * vecElemBytes;

    size_t shardVectors = (size_t)args.numVectors;
    if (vectorsMemoryLimit > 0) {
        shardVectors = vectorsMemoryLimit / vecBytesPer;
        if (shardVectors == 0) {
            shardVectors = 1;
        }
    }
    shardVectors = std::min(shardVectors, (size_t)args.numVectors);

    size_t shardQueries = (size_t)args.numQueries;
    if (queriesMemoryLimit > 0) {
        const size_t bytesPerQuery = (size_t)args.dims * sizeof(float) +
                shardVectors * sizeof(float);
        shardQueries = queriesMemoryLimit / bytesPerQuery;
        if (shardQueries == 0) {
            shardQueries = 1;
        }
    }
    shardQueries = std::min(shardQueries, (size_t)args.numQueries);

    std::vector<float> blockDist;
    for (idx_t qBase = 0; qBase < args.numQueries; qBase += (idx_t)shardQueries) {
        const idx_t nThisQ = std::min((idx_t)shardQueries, args.numQueries - qBase);
        const float* qPtr = queriesF32 + (size_t)qBase * args.dims;

        for (idx_t vBase = 0; vBase < args.numVectors; vBase += (idx_t)shardVectors) {
            const idx_t nThisV = std::min((idx_t)shardVectors, args.numVectors - vBase);

            blockDist.resize((size_t)nThisQ * nThisV);
            const float* normsPtr = vectorNorms ? (vectorNorms + (size_t)vBase) : nullptr;

            if (args.vectorType == MetalDistanceDataType::F16) {
                const uint16_t* vPtr = vectorsF16 + (size_t)vBase * args.dims;
                allPairsBlockFP16Vectors_(
                        resources,
                        vPtr,
                        nThisV,
                        qPtr,
                        nThisQ,
                        args.dims,
                        args.metric,
                        blockDist.data());
            } else {
                const float* vPtr = vectorsF32 + (size_t)vBase * args.dims;
                allPairsBlockF32_(
                        resources,
                        vPtr,
                        normsPtr,
                        nThisV,
                        qPtr,
                        nThisQ,
                        args.dims,
                        args.metric,
                        blockDist.data());
            }

            for (idx_t qi = 0; qi < nThisQ; ++qi) {
                float* dst = outDistances +
                        (size_t)(qBase + qi) * args.numVectors + (size_t)vBase;
                const float* src = blockDist.data() + (size_t)qi * nThisV;
                std::memcpy(dst, src, (size_t)nThisV * sizeof(float));
            }
        }
    }
}

const float* materializeQueryF32_(
        const MetalDistanceParams& args,
        std::vector<float>& convertedQueries) {
    const size_t numQueryElems = (size_t)args.numQueries * args.dims;
    if (args.queryType == MetalDistanceDataType::F32 && args.queriesRowMajor) {
        return static_cast<const float*>(args.queries);
    }
    convertedQueries.resize(numQueryElems);
    if (args.queryType == MetalDistanceDataType::F32 && !args.queriesRowMajor) {
        const float* q = static_cast<const float*>(args.queries);
        for (idx_t qi = 0; qi < args.numQueries; ++qi) {
            for (int d = 0; d < args.dims; ++d) {
                convertedQueries[(size_t)qi * args.dims + d] =
                        q[(size_t)d * args.numQueries + qi];
            }
        }
        return convertedQueries.data();
    }
    if (args.queryType == MetalDistanceDataType::F16 && args.queriesRowMajor) {
        const uint16_t* qh = static_cast<const uint16_t*>(args.queries);
        for (size_t i = 0; i < numQueryElems; ++i) {
            convertedQueries[i] = faiss::decode_fp16(qh[i]);
        }
        return convertedQueries.data();
    }
    if (args.queryType == MetalDistanceDataType::F16 && !args.queriesRowMajor) {
        const uint16_t* qh = static_cast<const uint16_t*>(args.queries);
        for (idx_t qi = 0; qi < args.numQueries; ++qi) {
            for (int d = 0; d < args.dims; ++d) {
                convertedQueries[(size_t)qi * args.dims + d] =
                        faiss::decode_fp16(qh[(size_t)d * args.numQueries + qi]);
            }
        }
        return convertedQueries.data();
    }
    if (args.queryType == MetalDistanceDataType::BF16 && args.queriesRowMajor) {
        const uint16_t* qh = static_cast<const uint16_t*>(args.queries);
        for (size_t i = 0; i < numQueryElems; ++i) {
            convertedQueries[i] = faiss::decode_bf16(qh[i]);
        }
        return convertedQueries.data();
    }
    if (args.queryType == MetalDistanceDataType::BF16 && !args.queriesRowMajor) {
        const uint16_t* qh = static_cast<const uint16_t*>(args.queries);
        for (idx_t qi = 0; qi < args.numQueries; ++qi) {
            for (int d = 0; d < args.dims; ++d) {
                convertedQueries[(size_t)qi * args.dims + d] =
                        faiss::decode_bf16(qh[(size_t)d * args.numQueries + qi]);
            }
        }
        return convertedQueries.data();
    }
    FAISS_THROW_MSG("bfKnn(params): unknown queryType");
    return nullptr;
}

const float* materializeVectorF32IfNeeded_(
        const MetalDistanceParams& args,
        std::vector<float>& convertedVectors) {
    if (args.vectorType == MetalDistanceDataType::F32 && args.vectorsRowMajor) {
        return static_cast<const float*>(args.vectors);
    }
    if (args.vectorType == MetalDistanceDataType::F32 && !args.vectorsRowMajor) {
        const float* v = static_cast<const float*>(args.vectors);
        const size_t numVectorElems = (size_t)args.numVectors * args.dims;
        convertedVectors.resize(numVectorElems);
        for (idx_t vi = 0; vi < args.numVectors; ++vi) {
            for (int d = 0; d < args.dims; ++d) {
                convertedVectors[(size_t)vi * args.dims + d] =
                        v[(size_t)d * args.numVectors + vi];
            }
        }
        return convertedVectors.data();
    }
    if (args.vectorType == MetalDistanceDataType::BF16) {
        const size_t numVectorElems = (size_t)args.numVectors * args.dims;
        const uint16_t* vh = static_cast<const uint16_t*>(args.vectors);
        convertedVectors.resize(numVectorElems);
        if (args.vectorsRowMajor) {
            for (size_t i = 0; i < numVectorElems; ++i) {
                convertedVectors[i] = faiss::decode_bf16(vh[i]);
            }
        } else {
            for (idx_t vi = 0; vi < args.numVectors; ++vi) {
                for (int d = 0; d < args.dims; ++d) {
                    convertedVectors[(size_t)vi * args.dims + d] =
                            faiss::decode_bf16(vh[(size_t)d * args.numVectors + vi]);
                }
            }
        }
        return convertedVectors.data();
    }
    return nullptr;
}

const uint16_t* materializeVectorF16IfNeeded_(
        const MetalDistanceParams& args,
        std::vector<uint16_t>& convertedVectors) {
    if (args.vectorType != MetalDistanceDataType::F16) {
        return nullptr;
    }
    if (args.vectorsRowMajor) {
        return static_cast<const uint16_t*>(args.vectors);
    }

    const size_t numVectorElems = (size_t)args.numVectors * args.dims;
    const uint16_t* vh = static_cast<const uint16_t*>(args.vectors);
    convertedVectors.resize(numVectorElems);
    for (idx_t vi = 0; vi < args.numVectors; ++vi) {
        for (int d = 0; d < args.dims; ++d) {
            convertedVectors[(size_t)vi * args.dims + d] =
                    vh[(size_t)d * args.numVectors + vi];
        }
    }
    return convertedVectors.data();
}

const float* materializeVectorF32FromF16_(
        const MetalDistanceParams& args,
        std::vector<float>& convertedVectorsF32) {
    if (args.vectorType != MetalDistanceDataType::F16) {
        return nullptr;
    }
    const uint16_t* v =
            static_cast<const uint16_t*>(args.vectors);
    const size_t n = (size_t)args.numVectors * args.dims;
    convertedVectorsF32.resize(n);
    if (args.vectorsRowMajor) {
        for (size_t i = 0; i < n; ++i) {
            convertedVectorsF32[i] = faiss::decode_fp16(v[i]);
        }
    } else {
        for (idx_t vi = 0; vi < args.numVectors; ++vi) {
            for (int d = 0; d < args.dims; ++d) {
                convertedVectorsF32[(size_t)vi * args.dims + d] =
                        faiss::decode_fp16(v[(size_t)d * args.numVectors + vi]);
            }
        }
    }
    return convertedVectorsF32.data();
}

void bfKnnCpuFallbackParams_(
        const MetalDistanceParams& args,
        const float* vectors,
        const float* queries,
        float* outDistances,
        void* outIndices,
        MetalIndicesDataType outIndicesType) {
    FAISS_THROW_IF_NOT(vectors && queries && outDistances);

    if (args.k == -1) {
        if (args.metric == METRIC_L2) {
            faiss::pairwise_L2sqr(
                    args.dims,
                    args.numQueries,
                    queries,
                    args.numVectors,
                    vectors,
                    outDistances);
        } else if (args.metric == METRIC_INNER_PRODUCT) {
            for (idx_t qi = 0; qi < args.numQueries; ++qi) {
                for (idx_t vi = 0; vi < args.numVectors; ++vi) {
                    float ip = 0.0f;
                    const float* q = queries + (size_t)qi * args.dims;
                    const float* v = vectors + (size_t)vi * args.dims;
                    for (int d = 0; d < args.dims; ++d) {
                        ip += q[d] * v[d];
                    }
                    outDistances[(size_t)qi * args.numVectors + vi] = ip;
                }
            }
        } else {
            faiss::pairwise_extra_distances(
                    args.dims,
                    args.numQueries,
                    queries,
                    args.numVectors,
                    vectors,
                    args.metric,
                    args.metricArg,
                    outDistances);
        }
        return;
    }

    FAISS_THROW_IF_NOT(outIndices);
    if (outIndicesType == MetalIndicesDataType::I64) {
        faiss::knn_extra_metrics(
                queries,
                vectors,
                args.dims,
                args.numQueries,
                args.numVectors,
                args.metric,
                args.metricArg,
                args.k,
                outDistances,
                static_cast<idx_t*>(outIndices));
        return;
    }

    if (outIndicesType == MetalIndicesDataType::I32) {
        std::vector<idx_t> tmpIdx((size_t)args.numQueries * args.k);
        faiss::knn_extra_metrics(
                queries,
                vectors,
                args.dims,
                args.numQueries,
                args.numVectors,
                args.metric,
                args.metricArg,
                args.k,
                outDistances,
                tmpIdx.data());
        int32_t* outIdx32 = static_cast<int32_t*>(outIndices);
        for (size_t i = 0; i < tmpIdx.size(); ++i) {
            outIdx32[i] = static_cast<int32_t>(tmpIdx[i]);
        }
        return;
    }

    FAISS_THROW_MSG("bfKnn(params): unknown outIndicesType in CPU fallback");
}

void bfKnn(
        std::shared_ptr<MetalResources> resources,
        const MetalDistanceParams& args) {
    FAISS_THROW_IF_NOT(args.vectors);
    FAISS_THROW_IF_NOT(args.queries);
    FAISS_THROW_IF_NOT_MSG(
            args.k > 0 || args.k == -1,
            "bfKnn(params): k must be > 0 or -1 for all-pairs");
    FAISS_THROW_IF_NOT(args.numVectors > 0);
    FAISS_THROW_IF_NOT(args.numQueries > 0);
    FAISS_THROW_IF_NOT(args.dims > 0);
    FAISS_THROW_IF_NOT(args.outDistances);
    if (args.k > 0) {
        FAISS_THROW_IF_NOT(args.outIndices);
    }

    std::vector<float> convertedQueries;
    const float* queries = materializeQueryF32_(args, convertedQueries);
    std::vector<float> convertedVectors;
    const float* vectorsF32 = materializeVectorF32IfNeeded_(args, convertedVectors);
    std::vector<uint16_t> convertedVectorsF16;
    const uint16_t* vectorsF16 =
            materializeVectorF16IfNeeded_(args, convertedVectorsF16);
    std::vector<float> convertedVectorsFromF16;
    const float* vectorsF32FromF16 =
            materializeVectorF32FromF16_(args, convertedVectorsFromF16);
    const float* vectorNormsF32 = args.vectorNorms;
    std::vector<float> convertedVectorNorms;
    if (args.vectorNorms && !args.vectorsRowMajor) {
        convertedVectorNorms.assign(args.vectorNorms, args.vectorNorms + args.numVectors);
        vectorNormsF32 = convertedVectorNorms.data();
    }

    if (args.k == -1) {
        FAISS_THROW_IF_NOT_MSG(
                !args.ignoreOutDistances,
                "bfKnn(params): ignoreOutDistances cannot be true when k == -1");
        if (!(args.metric == METRIC_L2 || args.metric == METRIC_INNER_PRODUCT)) {
            const float* vectorsForFallback =
                    (args.vectorType == MetalDistanceDataType::F16)
                    ? vectorsF32FromF16
                    : vectorsF32;
            bfKnnCpuFallbackParams_(
                    args,
                    vectorsForFallback,
                    queries,
                    args.outDistances,
                    nullptr,
                    args.outIndicesType);
            return;
        }
        allPairsDistanceTiledParams_(
                resources,
                args,
                vectorsF32,
                vectorsF16,
                queries,
                vectorNormsF32,
                0,
                0,
                args.outDistances);
        return;
    }

    if (!(args.metric == METRIC_L2 || args.metric == METRIC_INNER_PRODUCT)) {
        std::vector<float> tmpDistancesFallback;
        float* outDistFallback = args.outDistances;
        if (args.ignoreOutDistances) {
            tmpDistancesFallback.resize((size_t)args.numQueries * args.k);
            outDistFallback = tmpDistancesFallback.data();
        }
        const float* vectorsForFallback =
                (args.vectorType == MetalDistanceDataType::F16)
                ? vectorsF32FromF16
                : vectorsF32;
        bfKnnCpuFallbackParams_(
                args,
                vectorsForFallback,
                queries,
                outDistFallback,
                args.outIndices,
                args.outIndicesType);
        return;
    }

    const bool useVectorNorms =
            args.metric == METRIC_L2 && args.vectorNorms != nullptr;

    std::vector<float> tmpDistances;
    if (args.ignoreOutDistances) {
        tmpDistances.resize((size_t)args.numQueries * args.k);
    } else {
        FAISS_THROW_IF_NOT(args.outDistances);
    }
    float* outDist = args.ignoreOutDistances ? tmpDistances.data() : args.outDistances;

    if (args.outIndicesType == MetalIndicesDataType::I64) {
        idx_t* outIdx = static_cast<idx_t*>(args.outIndices);
        if (args.vectorType == MetalDistanceDataType::F16) {
            if (useVectorNorms) {
                bfKnnWithVectorNormsF32_(
                        resources,
                        vectorsF32FromF16,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        outIdx);
            } else {
                bfKnnFP16Vectors_(
                        resources,
                        vectorsF16,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        outIdx);
            }
        } else if (
                args.vectorType == MetalDistanceDataType::F32 ||
                args.vectorType == MetalDistanceDataType::BF16) {
            if (useVectorNorms) {
                bfKnnWithVectorNormsF32_(
                        resources,
                        vectorsF32,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        outIdx);
            } else {
                bfKnn(
                        resources,
                        vectorsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        outIdx);
            }
        } else {
            FAISS_THROW_MSG("bfKnn(params): unknown vectorType");
        }
        return;
    }

    if (args.outIndicesType == MetalIndicesDataType::I32) {
        std::vector<idx_t> tmpIdx((size_t)args.numQueries * args.k);
        if (args.vectorType == MetalDistanceDataType::F16) {
            if (useVectorNorms) {
                bfKnnWithVectorNormsF32_(
                        resources,
                        vectorsF32FromF16,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        tmpIdx.data());
            } else {
                bfKnnFP16Vectors_(
                        resources,
                        vectorsF16,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        tmpIdx.data());
            }
        } else if (
                args.vectorType == MetalDistanceDataType::F32 ||
                args.vectorType == MetalDistanceDataType::BF16) {
            if (useVectorNorms) {
                bfKnnWithVectorNormsF32_(
                        resources,
                        vectorsF32,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        tmpIdx.data());
            } else {
                bfKnn(
                        resources,
                        vectorsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        tmpIdx.data());
            }
        } else {
            FAISS_THROW_MSG("bfKnn(params): unknown vectorType");
        }
        int32_t* outIdx32 = static_cast<int32_t*>(args.outIndices);
        for (size_t i = 0; i < tmpIdx.size(); ++i) {
            outIdx32[i] = static_cast<int32_t>(tmpIdx[i]);
        }
        return;
    }

    FAISS_THROW_MSG("bfKnn(params): unknown outIndicesType");
}

void bfKnn_tiling(
        std::shared_ptr<MetalResources> resources,
        const MetalDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
    FAISS_THROW_IF_NOT(args.vectors);
    FAISS_THROW_IF_NOT(args.queries);
    FAISS_THROW_IF_NOT_MSG(
            args.k > 0 || args.k == -1,
            "bfKnn_tiling(params): k must be > 0 or -1 for all-pairs");
    FAISS_THROW_IF_NOT(args.numVectors > 0);
    FAISS_THROW_IF_NOT(args.numQueries > 0);
    FAISS_THROW_IF_NOT(args.dims > 0);
    FAISS_THROW_IF_NOT(args.outDistances);
    if (args.k > 0) {
        FAISS_THROW_IF_NOT(args.outIndices);
    }

    std::vector<float> convertedQueries;
    const float* queries = materializeQueryF32_(args, convertedQueries);
    std::vector<float> convertedVectors;
    const float* vectorsF32 = materializeVectorF32IfNeeded_(args, convertedVectors);
    std::vector<uint16_t> convertedVectorsF16;
    const uint16_t* vectorsF16 =
            materializeVectorF16IfNeeded_(args, convertedVectorsF16);
    std::vector<float> convertedVectorsFromF16;
    const float* vectorsF32FromF16 =
            materializeVectorF32FromF16_(args, convertedVectorsFromF16);
    const float* vectorNormsF32 = args.vectorNorms;
    std::vector<float> convertedVectorNorms;
    if (args.vectorNorms && !args.vectorsRowMajor) {
        convertedVectorNorms.assign(args.vectorNorms, args.vectorNorms + args.numVectors);
        vectorNormsF32 = convertedVectorNorms.data();
    }

    if (args.k == -1) {
        FAISS_THROW_IF_NOT_MSG(
                !args.ignoreOutDistances,
                "bfKnn_tiling(params): ignoreOutDistances cannot be true when k == -1");
        if (!(args.metric == METRIC_L2 || args.metric == METRIC_INNER_PRODUCT)) {
            const float* vectorsForFallback =
                    (args.vectorType == MetalDistanceDataType::F16)
                    ? vectorsF32FromF16
                    : vectorsF32;
            bfKnnCpuFallbackParams_(
                    args,
                    vectorsForFallback,
                    queries,
                    args.outDistances,
                    nullptr,
                    args.outIndicesType);
            return;
        }
        allPairsDistanceTiledParams_(
                resources,
                args,
                vectorsF32,
                vectorsF16,
                queries,
                vectorNormsF32,
                vectorsMemoryLimit,
                queriesMemoryLimit,
                args.outDistances);
        return;
    }

    if (!(args.metric == METRIC_L2 || args.metric == METRIC_INNER_PRODUCT)) {
        std::vector<float> tmpDistancesFallback;
        float* outDistFallback = args.outDistances;
        if (args.ignoreOutDistances) {
            tmpDistancesFallback.resize((size_t)args.numQueries * args.k);
            outDistFallback = tmpDistancesFallback.data();
        }
        const float* vectorsForFallback =
                (args.vectorType == MetalDistanceDataType::F16)
                ? vectorsF32FromF16
                : vectorsF32;
        bfKnnCpuFallbackParams_(
                args,
                vectorsForFallback,
                queries,
                outDistFallback,
                args.outIndices,
                args.outIndicesType);
        return;
    }

    const bool useVectorNorms =
            args.metric == METRIC_L2 && args.vectorNorms != nullptr;

    std::vector<float> tmpDistances;
    if (args.ignoreOutDistances) {
        tmpDistances.resize((size_t)args.numQueries * args.k);
    } else {
        FAISS_THROW_IF_NOT(args.outDistances);
    }
    float* outDist = args.ignoreOutDistances ? tmpDistances.data() : args.outDistances;

    if (args.outIndicesType == MetalIndicesDataType::I64) {
        idx_t* outIdx = static_cast<idx_t*>(args.outIndices);
        if (args.vectorType == MetalDistanceDataType::F16) {
            if (useVectorNorms) {
                bfKnn_tilingWithNormsF32_(
                        resources,
                        vectorsF32FromF16,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        outIdx,
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            } else {
                bfKnn_tilingFP16_(
                        resources,
                        vectorsF16,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        outIdx,
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            }
        } else if (
                args.vectorType == MetalDistanceDataType::F32 ||
                args.vectorType == MetalDistanceDataType::BF16) {
            if (useVectorNorms) {
                bfKnn_tilingWithNormsF32_(
                        resources,
                        vectorsF32,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        outIdx,
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            } else {
                bfKnn_tiling(
                        resources,
                        vectorsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        outIdx,
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            }
        } else {
            FAISS_THROW_MSG("bfKnn_tiling(params): unknown vectorType");
        }
        return;
    }

    if (args.outIndicesType == MetalIndicesDataType::I32) {
        std::vector<idx_t> tmpIdx((size_t)args.numQueries * args.k);
        if (args.vectorType == MetalDistanceDataType::F16) {
            if (useVectorNorms) {
                bfKnn_tilingWithNormsF32_(
                        resources,
                        vectorsF32FromF16,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        tmpIdx.data(),
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            } else {
                bfKnn_tilingFP16_(
                        resources,
                        vectorsF16,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        tmpIdx.data(),
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            }
        } else if (
                args.vectorType == MetalDistanceDataType::F32 ||
                args.vectorType == MetalDistanceDataType::BF16) {
            if (useVectorNorms) {
                bfKnn_tilingWithNormsF32_(
                        resources,
                        vectorsF32,
                        vectorNormsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        outDist,
                        tmpIdx.data(),
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            } else {
                bfKnn_tiling(
                        resources,
                        vectorsF32,
                        args.numVectors,
                        queries,
                        args.numQueries,
                        args.dims,
                        args.k,
                        args.metric,
                        outDist,
                        tmpIdx.data(),
                        vectorsMemoryLimit,
                        queriesMemoryLimit);
            }
        } else {
            FAISS_THROW_MSG("bfKnn_tiling(params): unknown vectorType");
        }
        int32_t* outIdx32 = static_cast<int32_t*>(args.outIndices);
        for (size_t i = 0; i < tmpIdx.size(); ++i) {
            outIdx32[i] = static_cast<int32_t>(tmpIdx[i]);
        }
        return;
    }

    FAISS_THROW_MSG("bfKnn_tiling(params): unknown outIndicesType");
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
    bool useIL = (interleavedCodes != nil && interleavedCodesOffset != nil);
    if (!device || !queue || !queries || (!codes && !useIL) || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

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
        id<MTLBuffer> centroids,
        bool byResidual,
        id<MTLBuffer> outDistances, id<MTLBuffer> outIndices,
        id<MTLBuffer> perListDistBuf, id<MTLBuffer> perListIdxBuf) {
    if (!device || !queue || !queries || !codes || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0) return false;
    if (sqType != MetalSQType::FP16 &&
        sqType != MetalSQType::SQ8_DIRECT &&
        !sqTables) {
        return false;
    }
    if (byResidual && !centroids) {
        return false;
    }

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    IVFScanVariant variant = IVFScanVariant::FP16;
    switch (sqType) {
        case MetalSQType::SQ4:
            variant = IVFScanVariant::SQ4;
            break;
        case MetalSQType::SQ6:
            variant = IVFScanVariant::SQ6;
            break;
        case MetalSQType::SQ8:
            variant = IVFScanVariant::SQ8;
            break;
        case MetalSQType::SQ8_DIRECT:
            variant = IVFScanVariant::SQ8Direct;
            break;
        case MetalSQType::FP16:
            variant = IVFScanVariant::FP16;
            break;
    }

    uint32_t sp[6] = {(uint32_t)nq, (uint32_t)d, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u,
                      byResidual ? 1u : 0u};
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
                        sqTables,
                        centroids);
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
        int nq, int M, int k, int nprobe, int avgListLen, bool lookupFp16, bool isL2,
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

    bool useSmall = (!lookupFp16 && avgListLen > 0 && avgListLen <= 32 && k <= 32);
    K.encodeIVFPQScanList(enc, useSmall, lookupFp16, lookupTable, codes, ids,
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
        id<MTLBuffer> outLookup) {
    if (!device || !queue || !queries || !coarseAssign || !pqCentroids ||
        !outLookup) {
        return false;
    }
    if (nq <= 0 || d <= 0 || M <= 0 || nprobe <= 0) return false;
    if (isL2 && !coarseCentroids) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    K.encodeIVFPQBuildLookupTables(
            enc,
            isL2,
            lookupFp16,
            queries,
            coarseAssign,
            coarseCentroids,
            pqCentroids,
            outLookup,
            nq,
            d,
            M,
            nprobe);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

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
        id<MTLBuffer> perListIdxBuf) {
    if (!device || !queue || !queries || !coarseAssign || !pqCentroids ||
        !lookupTable || !codes || !ids || !listOffset || !listLength ||
        !outDistances || !outIndices || !perListDistBuf || !perListIdxBuf) {
        return false;
    }
    if (nq <= 0 || d <= 0 || M <= 0 || k <= 0 || nprobe <= 0) return false;
    if (isL2 && !coarseCentroids) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    uint32_t sp[5] = {(uint32_t)nq, (uint32_t)M, (uint32_t)k,
                      (uint32_t)nprobe, isL2 ? 1u : 0u};
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:sp length:sizeof(sp)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    K.encodeIVFPQBuildLookupTables(
            enc,
            isL2,
            lookupFp16,
            queries,
            coarseAssign,
            coarseCentroids,
            pqCentroids,
            lookupTable,
            nq,
            d,
            M,
            nprobe);

    bool useSmall = (!lookupFp16 && avgListLen > 0 && avgListLen <= 32 && k <= 32);
    K.encodeIVFPQScanList(
            enc,
            useSmall,
            lookupFp16,
            lookupTable,
            codes,
            ids,
            listOffset,
            listLength,
            coarseAssign,
            perListDistBuf,
            perListIdxBuf,
            paramsBuf,
            nq,
            nprobe);

    K.encodeIVFMergeLists(
            enc,
            perListDistBuf,
            perListIdxBuf,
            outDistances,
            outIndices,
            paramsBuf,
            nq);

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

bool runMetalConvertF32ToF16(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> srcF32,
        id<MTLBuffer> dstF16,
        size_t numElems) {
    if (!device || !queue || !srcF32 || !dstF16 || numElems == 0) return false;
    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    K.encodeConvertF32ToF16(enc, srcF32, dstF16, numElems);
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
    bool useIL = (interleavedCodes != nil && interleavedCodesOffset != nil);
    if (!device || !queue || !queries || !centroids || (!codes && !useIL) || !ids ||
        !listOffset || !listLength || !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf ||
        !coarseDistBuf || !coarseIdxBuf || !distMatrixBuf)
        return false;
    if (k <= 0 || nq <= 0 || nprobe <= 0 || nlist <= 0) return false;

    MetalKernels& K = getMetalKernels(device);
    if (!K.isValid()) return false;

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
