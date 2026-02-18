// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Reusable distance computation for Metal backend.
 * Mirrors faiss/gpu/impl/Distance.cu for CUDA.
 */

#import "MetalDistance.h"
#import "MetalResources.h"
#include <algorithm>

namespace faiss {
namespace gpu_metal {

namespace {

// Embedded MSL source as fallback (if MetalDistance.metal file can't be loaded)
static const char* kMSLSourceEmbedded = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void l2_squared_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;
    float sum = 0.0f;
    for (uint t = 0; t < d; t++) {
        float a = queries[i * d + t];
        float b = vectors[j * d + t];
        sum += (a - b) * (a - b);
    }
    distances[i * nb + j] = sum;
}

kernel void ip_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;
    float sum = 0.0f;
    for (uint t = 0; t < d; t++)
        sum += queries[i * d + t] * vectors[j * d + t];
    distances[i * nb + j] = sum;
}

#define TOPK_HEAP_VARIANT(K) \
kernel void topk_heap_##K( \
    device const float* distances [[buffer(0)]], \
    device float* outDistances [[buffer(1)]], \
    device int* outIndices [[buffer(2)]], \
    device const uint* params [[buffer(3)]], \
    uint qi [[thread_position_in_grid]] \
) { \
    threadgroup float smemK[K]; \
    threadgroup int smemIdx[K]; \
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    const device float* row = distances + qi * nb; \
    uint kk = min(k, nb); \
    uint n = kk; \
    for (uint i = 0; i < n; i++) { smemK[i] = row[i]; smemIdx[i] = (int)i; } \
    for (uint i = 0; i < n; i++) { \
        for (uint j = i + 1; j < n; j++) { \
            bool swap = want_min ? (smemK[j] < smemK[i]) : (smemK[j] > smemK[i]); \
            if (swap) { float td = smemK[i]; smemK[i] = smemK[j]; smemK[j] = td; \
                        int ti = smemIdx[i]; smemIdx[i] = smemIdx[j]; smemIdx[j] = ti; } \
        } \
    } \
    for (uint i = n; i < (uint)K; i++) { smemK[i] = want_min ? 1e38f : -1e38f; smemIdx[i] = -1; } \
    for (uint j = kk; j < nb; j++) { \
        float v = row[j]; \
        bool insert = want_min ? (v < smemK[kk-1]) : (v > smemK[kk-1]); \
        if (!insert) continue; \
        uint pos = kk - 1; \
        if (want_min) { \
            while (pos > 0 && v < smemK[pos-1]) { smemK[pos] = smemK[pos-1]; smemIdx[pos] = smemIdx[pos-1]; pos--; } \
        } else { \
            while (pos > 0 && v > smemK[pos-1]) { smemK[pos] = smemK[pos-1]; smemIdx[pos] = smemIdx[pos-1]; pos--; } \
        } \
        smemK[pos] = v; smemIdx[pos] = (int)j; \
    } \
    for (uint i = 0; i < kk; i++) { outDistances[qi * k + i] = smemK[i]; outIndices[qi * k + i] = smemIdx[i]; } \
    for (uint i = kk; i < k; i++) { outDistances[qi * k + i] = want_min ? 1e38f : -1e38f; outIndices[qi * k + i] = -1; } \
}
TOPK_HEAP_VARIANT(32)
TOPK_HEAP_VARIANT(64)
TOPK_HEAP_VARIANT(128)
TOPK_HEAP_VARIANT(256)
TOPK_HEAP_VARIANT(512)
TOPK_HEAP_VARIANT(1024)
TOPK_HEAP_VARIANT(2048)
#undef TOPK_HEAP_VARIANT

#define TOPK_MERGE_VARIANT(K) \
kernel void topk_merge_pair_##K( \
    device const float* inK [[buffer(0)]], \
    device const int* inV [[buffer(1)]], \
    device float* outK [[buffer(2)]], \
    device int* outIdx [[buffer(3)]], \
    device const uint* params [[buffer(4)]], \
    uint qi [[thread_position_in_grid]] \
) { \
    threadgroup float smemK[K]; \
    threadgroup int smemIdx[K]; \
    uint nq = params[0], numTiles = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    uint kk = min(k, (uint)K); \
    for (uint i = 0; i < kk; i++) { smemK[i] = want_min ? 1e38f : -1e38f; smemIdx[i] = -1; } \
    uint totalCandidates = numTiles * k; \
    for (uint tile = 0; tile < numTiles; tile++) { \
        for (uint i = 0; i < k; i++) { \
            uint idx = qi * totalCandidates + tile * k + i; \
            float dist = inK[idx]; \
            int vidx = inV[idx]; \
            if (vidx < 0) continue; \
            bool insert = want_min ? (dist < smemK[kk-1]) : (dist > smemK[kk-1]); \
            if (!insert) continue; \
            uint pos = kk - 1; \
            if (want_min) { \
                while (pos > 0 && dist < smemK[pos-1]) { smemK[pos] = smemK[pos-1]; smemIdx[pos] = smemIdx[pos-1]; pos--; } \
            } else { \
                while (pos > 0 && dist > smemK[pos-1]) { smemK[pos] = smemK[pos-1]; smemIdx[pos] = smemIdx[pos-1]; pos--; } \
            } \
            smemK[pos] = dist; smemIdx[pos] = vidx; \
        } \
    } \
    for (uint i = 0; i < kk; i++) { outK[qi * k + i] = smemK[i]; outIdx[qi * k + i] = smemIdx[i]; } \
    for (uint i = kk; i < k; i++) { outK[qi * k + i] = want_min ? 1e38f : -1e38f; outIdx[qi * k + i] = -1; } \
}
TOPK_MERGE_VARIANT(32)
TOPK_MERGE_VARIANT(64)
TOPK_MERGE_VARIANT(128)
TOPK_MERGE_VARIANT(256)
TOPK_MERGE_VARIANT(512)
TOPK_MERGE_VARIANT(1024)
TOPK_MERGE_VARIANT(2048)
#undef TOPK_MERGE_VARIANT

kernel void increment_index(
    device int* indices [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], k = params[1], tileCols = params[2], numTiles = params[3];
    uint qi = gid.y;
    uint tileIdx = gid.x;
    if (qi >= nq || tileIdx >= numTiles) return;
    uint offset = tileIdx * tileCols;
    uint baseIdx = qi * numTiles * k + tileIdx * k;
    for (uint i = 0; i < k; i++) {
        int vidx = indices[baseIdx + i];
        if (vidx >= 0) {
            indices[baseIdx + i] = vidx + (int)offset;
        }
    }
}
)msl";

// Load MSL kernel source from MetalDistance.metal file, with fallback to embedded source
static NSString* loadMSLSource() {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* metalPath = nil;
    
    // Try multiple paths:
    // 1. Try in main bundle (for app bundles)
    NSBundle* mainBundle = [NSBundle mainBundle];
    metalPath = [mainBundle pathForResource:@"MetalDistance" ofType:@"metal"];
    
    // 2. Try relative to current working directory (for tests/build from repo root)
    if (!metalPath) {
        NSString* cwd = [fm currentDirectoryPath];
        NSString* relPath = [cwd stringByAppendingPathComponent:@"faiss/gpu_metal/MetalDistance.metal"];
        if ([fm fileExistsAtPath:relPath]) {
            metalPath = relPath;
        }
    }
    
    // 3. Try relative to executable (for installed libraries)
    if (!metalPath) {
        NSString* execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* relPath = [execDir stringByAppendingPathComponent:@"MetalDistance.metal"];
            if ([fm fileExistsAtPath:relPath]) {
                metalPath = relPath;
            }
        }
    }
    
    // If file found, load it
    if (metalPath) {
        NSError* err = nil;
        NSString* source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
        if (source && !err) {
            return source;
        }
    }
    
    // Fallback to embedded source
    return @(kMSLSourceEmbedded);
}

// Maximum k supported (fits in 16 KB threadgroup memory: 8*k bytes).
static constexpr int kMaxK = 2048;

// Variant sizes: pick smallest K >= k. Order must match kTopKVariantNames.
static const int kTopKVariantSizes[] = {32, 64, 128, 256, 512, 1024, 2048};
static const int kNumTopKVariants = sizeof(kTopKVariantSizes) / sizeof(kTopKVariantSizes[0]);
static const char* kTopKVariantNames[] = {
    "topk_heap_32", "topk_heap_64", "topk_heap_128", "topk_heap_256",
    "topk_heap_512", "topk_heap_1024", "topk_heap_2048"
};
static const char* kMergeVariantNames[] = {
    "topk_merge_pair_32", "topk_merge_pair_64", "topk_merge_pair_128", "topk_merge_pair_256",
    "topk_merge_pair_512", "topk_merge_pair_1024", "topk_merge_pair_2048"
};

static int selectTopKVariant(int k) {
    for (int i = 0; i < kNumTopKVariants; i++) {
        if (k <= kTopKVariantSizes[i]) {
            return i;
        }
    }
    return kNumTopKVariants - 1;
}

} // namespace

void chooseTileSize(
        int nq,
        int nb,
        int d,
        size_t elementSize,
        size_t availableMem,
        int& tileRows,
        int& tileCols) {
    // Target: ~512 MB per tile (M-series typically 8-16 GB unified memory)
    // Divide by 2 for double-buffering
    size_t targetUsage = 512 * 1024 * 1024;
    targetUsage /= 2;
    targetUsage /= elementSize;
    
    // Preferred tileRows: 512 for float32, 1024 if dim <= 32
    int preferredTileRows = 512;
    if (d <= 32) {
        preferredTileRows = 1024;
    }
    
    tileRows = std::min(preferredTileRows, nq);
    tileCols = std::min((int)(targetUsage / preferredTileRows), nb);
    
    // Ensure minimum sizes
    if (tileRows < 1) tileRows = 1;
    if (tileCols < 1) tileCols = 1;
}

int getMetalDistanceMaxK() {
    return kMaxK;
}

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
        id<MTLBuffer> outIndices) {
    if (!device || !queue || !queries || !vectors || !outDistances || !outIndices) {
        return false;
    }
    if (k <= 0 || k > kMaxK) {
        return false;
    }

    // Load MSL source from MetalDistance.metal file
    NSString* mslSource = loadMSLSource();
    if (!mslSource) {
        // Fallback: if file loading fails, return error
        // In production, you might want to embed the source as a fallback
        return false;
    }
    
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:mslSource options:nil error:&err];
    if (!lib) {
        return false;
    }

    // Calculate tile sizes
    int tileRows, tileCols;
    size_t availableMem = 512 * 1024 * 1024;  // TODO: query from MetalResources if available
    chooseTileSize(nq, nb, d, sizeof(float), availableMem, tileRows, tileCols);
    
    bool needsTiling = (tileCols < nb || tileRows < nq);
    
    id<MTLFunction> fnDist = [lib newFunctionWithName:isL2 ? @"l2_squared_matrix" : @"ip_matrix"];
    int variantIndex = selectTopKVariant(k);
    id<MTLFunction> fnTopK = [lib newFunctionWithName:@(kTopKVariantNames[variantIndex])];
    id<MTLFunction> fnMerge = nil;
    id<MTLFunction> fnIncrement = nil;
    if (needsTiling) {
        fnMerge = [lib newFunctionWithName:@(kMergeVariantNames[variantIndex])];
        fnIncrement = [lib newFunctionWithName:@"increment_index"];
    }
    
    if (!fnDist || !fnTopK || (needsTiling && (!fnMerge || !fnIncrement))) {
        return false;
    }

    id<MTLComputePipelineState> psDist = [device newComputePipelineStateWithFunction:fnDist error:&err];
    id<MTLComputePipelineState> psTopK = [device newComputePipelineStateWithFunction:fnTopK error:&err];
    id<MTLComputePipelineState> psMerge = nil;
    id<MTLComputePipelineState> psIncrement = nil;
    if (needsTiling) {
        psMerge = [device newComputePipelineStateWithFunction:fnMerge error:&err];
        psIncrement = [device newComputePipelineStateWithFunction:fnIncrement error:&err];
        if (!psMerge || !psIncrement) {
            return false;
        }
    }
    if (!psDist || !psTopK) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    const NSUInteger w = 16;
    const NSUInteger h = 16;

    if (!needsTiling) {
        // Single-pass: compute full distance matrix, then top-k
        [enc setComputePipelineState:psDist];
        [enc setBuffer:queries offset:0 atIndex:0];
        [enc setBuffer:vectors offset:0 atIndex:1];
        id<MTLBuffer> distMatrix = [device newBufferWithLength:(size_t)nq * (size_t)nb * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        if (!distMatrix) {
            [enc endEncoding];
            return false;
        }
        [enc setBuffer:distMatrix offset:0 atIndex:2];
        uint32_t distArgs[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
        [enc setBytes:distArgs length:sizeof(distArgs) atIndex:3];
        MTLSize tgSize = MTLSizeMake(w, h, 1);
        MTLSize gridSize = MTLSizeMake((nb + w - 1) / w, (nq + h - 1) / h, 1);
        [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

        [enc setComputePipelineState:psTopK];
        [enc setBuffer:distMatrix offset:0 atIndex:0];
        [enc setBuffer:outDistances offset:0 atIndex:1];
        [enc setBuffer:outIndices offset:0 atIndex:2];
        uint32_t topkArgs[4] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)k, isL2 ? 1u : 0u};
        [enc setBytes:topkArgs length:sizeof(topkArgs) atIndex:3];
        MTLSize gridTopK = MTLSizeMake(nq, 1, 1);
        [enc dispatchThreadgroups:gridTopK threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    } else {
        // Two-level tiling: tile over queries and vectors
        int numRowTiles = (nq + tileRows - 1) / tileRows;
        int numColTiles = (nb + tileCols - 1) / tileCols;
        
        // Temporary buffers for per-tile top-k results: (nq, numColTiles * k)
        id<MTLBuffer> outDistanceBuf = [device newBufferWithLength:
            (size_t)nq * (size_t)numColTiles * (size_t)k * sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIndexBuf = [device newBufferWithLength:
            (size_t)nq * (size_t)numColTiles * (size_t)k * sizeof(int32_t)
            options:MTLResourceStorageModeShared];
        if (!outDistanceBuf || !outIndexBuf) {
            [enc endEncoding];
            return false;
        }
        
        // Tile over queries
        for (int tileRow = 0; tileRow < numRowTiles; tileRow++) {
            int curQuerySize = std::min(tileRows, nq - tileRow * tileRows);
            size_t queryBaseOffset = (size_t)(tileRow * tileRows) * (size_t)numColTiles * (size_t)k;
            
            // Tile over vectors
            for (int tileCol = 0; tileCol < numColTiles; tileCol++) {
                int curVecSize = std::min(tileCols, nb - tileCol * tileCols);
                
                // Compute distance matrix for this tile
                [enc setComputePipelineState:psDist];
                [enc setBuffer:queries offset:(size_t)(tileRow * tileRows * d) * sizeof(float) atIndex:0];
                [enc setBuffer:vectors offset:(size_t)(tileCol * tileCols * d) * sizeof(float) atIndex:1];
                id<MTLBuffer> distMatrixTile = [device newBufferWithLength:
                    (size_t)curQuerySize * (size_t)curVecSize * sizeof(float)
                    options:MTLResourceStorageModeShared];
                if (!distMatrixTile) {
                    [enc endEncoding];
                    return false;
                }
                [enc setBuffer:distMatrixTile offset:0 atIndex:2];
                uint32_t distArgs[3] = {(uint32_t)curQuerySize, (uint32_t)curVecSize, (uint32_t)d};
                [enc setBytes:distArgs length:sizeof(distArgs) atIndex:3];
                MTLSize tgSize = MTLSizeMake(w, h, 1);
                MTLSize gridSize = MTLSizeMake((curVecSize + w - 1) / w, (curQuerySize + h - 1) / h, 1);
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
                
                // Top-k for this tile: output directly to main buffer
                // Layout: [queryIdx][tileCol][k] = queryIdx * (numColTiles * k) + tileCol * k
                [enc setComputePipelineState:psTopK];
                [enc setBuffer:distMatrixTile offset:0 atIndex:0];
                // Output offset for this tile: queryBaseOffset + queryIdx * (numColTiles * k) + tileCol * k
                // But top-k kernel expects contiguous output per query, so we need temp buffers
                id<MTLBuffer> tileOutDist = [device newBufferWithLength:
                    (size_t)curQuerySize * (size_t)k * sizeof(float)
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> tileOutIdx = [device newBufferWithLength:
                    (size_t)curQuerySize * (size_t)k * sizeof(int32_t)
                    options:MTLResourceStorageModeShared];
                if (!tileOutDist || !tileOutIdx) {
                    [enc endEncoding];
                    return false;
                }
                [enc setBuffer:tileOutDist offset:0 atIndex:1];
                [enc setBuffer:tileOutIdx offset:0 atIndex:2];
                uint32_t topkArgs[4] = {(uint32_t)curQuerySize, (uint32_t)curVecSize, (uint32_t)k, isL2 ? 1u : 0u};
                [enc setBytes:topkArgs length:sizeof(topkArgs) atIndex:3];
                MTLSize gridTopK = MTLSizeMake(curQuerySize, 1, 1);
                [enc dispatchThreadgroups:gridTopK threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                
                // Copy tile results to main buffer (interleaved by tile column)
                // Use blit encoder for efficient copy
                [enc endEncoding];
                id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                for (int q = 0; q < curQuerySize; q++) {
                    size_t srcOffset = (size_t)q * (size_t)k;
                    size_t dstOffset = queryBaseOffset + (size_t)q * (size_t)numColTiles * (size_t)k + (size_t)tileCol * (size_t)k;
                    [blitEnc copyFromBuffer:tileOutDist sourceOffset:srcOffset * sizeof(float)
                                 toBuffer:outDistanceBuf destinationOffset:dstOffset * sizeof(float)
                                 size:k * sizeof(float)];
                    [blitEnc copyFromBuffer:tileOutIdx sourceOffset:srcOffset * sizeof(int32_t)
                                 toBuffer:outIndexBuf destinationOffset:dstOffset * sizeof(int32_t)
                                 size:k * sizeof(int32_t)];
                }
                [blitEnc endEncoding];
                enc = [cmdBuf computeCommandEncoder];
            }
            
            // After all vector tiles for this query batch: adjust indices and merge
            // Adjust indices: add tileCols * tileCol to make them global
            if (numColTiles > 1) {
                [enc setComputePipelineState:psIncrement];
                [enc setBuffer:outIndexBuf offset:queryBaseOffset * sizeof(int32_t) atIndex:0];
                uint32_t incArgs[4] = {(uint32_t)curQuerySize, (uint32_t)k, (uint32_t)tileCols, (uint32_t)numColTiles};
                [enc setBytes:incArgs length:sizeof(incArgs) atIndex:1];
                // Grid: (numColTiles, curQuerySize) - one threadgroup per (tile, query) pair
                MTLSize gridInc = MTLSizeMake(numColTiles, curQuerySize, 1);
                [enc dispatchThreadgroups:gridInc threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
            
            // Merge per-tile top-k lists into final top-k
            if (numColTiles > 1) {
                [enc setComputePipelineState:psMerge];
                [enc setBuffer:outDistanceBuf offset:queryBaseOffset * sizeof(float) atIndex:0];
                [enc setBuffer:outIndexBuf offset:queryBaseOffset * sizeof(int32_t) atIndex:1];
                [enc setBuffer:outDistances offset:(size_t)(tileRow * tileRows * k) * sizeof(float) atIndex:2];
                [enc setBuffer:outIndices offset:(size_t)(tileRow * tileRows * k) * sizeof(int32_t) atIndex:3];
                uint32_t mergeArgs[4] = {(uint32_t)curQuerySize, (uint32_t)numColTiles, (uint32_t)k, isL2 ? 1u : 0u};
                [enc setBytes:mergeArgs length:sizeof(mergeArgs) atIndex:4];
                MTLSize gridMerge = MTLSizeMake(curQuerySize, 1, 1);
                [enc dispatchThreadgroups:gridMerge threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } else {
                // Single column tile: copy directly (no merge needed)
                [enc endEncoding];
                id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                size_t srcOffset = (size_t)(tileRow * tileRows) * (size_t)k;
                size_t dstOffset = srcOffset;
                [blitEnc copyFromBuffer:outDistanceBuf sourceOffset:srcOffset * sizeof(float)
                             toBuffer:outDistances destinationOffset:dstOffset * sizeof(float)
                             size:curQuerySize * k * sizeof(float)];
                [blitEnc copyFromBuffer:outIndexBuf sourceOffset:srcOffset * sizeof(int32_t)
                             toBuffer:outIndices destinationOffset:dstOffset * sizeof(int32_t)
                             size:curQuerySize * k * sizeof(int32_t)];
                [blitEnc endEncoding];
                enc = [cmdBuf computeCommandEncoder];
            }
        }
    }

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return true;
}

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
        id<MTLBuffer> outIndices) {
    return runMetalDistance(device, queue, queries, vectors, nq, nb, d, k, true, outDistances, outIndices);
}

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
        id<MTLBuffer> outIndices) {
    return runMetalDistance(device, queue, queries, vectors, nq, nb, d, k, false, outDistances, outIndices);
}

} // namespace gpu_metal
} // namespace faiss
