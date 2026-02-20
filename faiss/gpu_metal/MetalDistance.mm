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

// Legacy heap-based top-k variants: one thread per query, heap in threadgroup memory.
// LEGACY: Used for non-tiled path (full matrix) and as fallback when parallel kernels unavailable.
// For tiled path, prefer simdgroupSelect (K′≤64) or threadgroupSelect (K′≤1024).
// Variants: K in {32,64,128,256,512,1024,2048}.
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

#define SIMDGROUP_SELECT_VARIANT(K, R_PER_LANE) \
kernel void topk_simdgroup_##K( \
    device const float* distances [[buffer(0)]], \
    device float* outDistances [[buffer(1)]], \
    device int* outIndices [[buffer(2)]], \
    device const uint* params [[buffer(3)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint SIMD_WIDTH = 32; \
    constexpr uint R = R_PER_LANE; \
    constexpr uint MAX_CANDIDATES = SIMD_WIDTH * R; \
    threadgroup float tgDist[MAX_CANDIDATES]; \
    threadgroup int tgIdx[MAX_CANDIDATES]; \
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    /* Only first SIMD_WIDTH threads participate (one simdgroup per query) */ \
    if (tid >= SIMD_WIDTH) return; \
    uint lane_id = tid; \
    const device float* row = distances + qi * nb; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    \
    /* Per-lane: strided scan, keep local top-R in registers */ \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    \
    for (uint j = lane_id; j < nb; j += SIMD_WIDTH) { \
        float v = row[j]; \
        /* Insert into local sorted list (keep best R) */ \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; \
                localIdx[pos] = localIdx[pos-1]; \
                pos--; \
            } \
            localDist[pos] = v; \
            localIdx[pos] = (int)j; \
            localCount++; \
        } else { \
            bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; \
                    localIdx[pos] = localIdx[pos-1]; \
                    pos--; \
                } \
                localDist[pos] = v; \
                localIdx[pos] = (int)j; \
            } \
        } \
    } \
    \
    /* Write local candidates to threadgroup memory */ \
    for (uint i = 0; i < R; i++) { \
        uint idx = lane_id * R + i; \
        if (i < localCount) { \
            tgDist[idx] = localDist[i]; \
            tgIdx[idx] = localIdx[i]; \
        } else { \
            tgDist[idx] = want_min ? 1e38f : -1e38f; \
            tgIdx[idx] = -1; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Bitonic merge across lanes using simdgroup operations */ \
    /* For k=32 with R=1: we have 32 candidates, need top-k */ \
    /* For k=64 with R=2: we have 64 candidates, need top-k */ \
    uint activeSize = min(MAX_CANDIDATES, nb); \
    uint numPasses = min(activeSize, 16u); \
    for (uint pass = 0; pass < numPasses; pass++) { \
        uint idx = lane_id; \
        if (idx >= activeSize - 1) { \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            continue; \
        } \
        /* Compare-swap: even indices with next on even passes, odd on odd passes */ \
        bool doCompare = (pass % 2 == 0) ? (idx % 2 == 0) : (idx % 2 == 1); \
        if (doCompare && idx + 1 < activeSize) { \
            bool swap = want_min ? (tgDist[idx + 1] < tgDist[idx]) : (tgDist[idx + 1] > tgDist[idx]); \
            if (swap) { \
                float td = tgDist[idx]; tgDist[idx] = tgDist[idx + 1]; tgDist[idx + 1] = td; \
                int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[idx + 1]; tgIdx[idx + 1] = ti; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    /* Output: first K_out lanes write results (sorted) */ \
    if (lane_id < K_out) { \
        outDistances[qi * k + lane_id] = tgDist[lane_id]; \
        outIndices[qi * k + lane_id] = tgIdx[lane_id]; \
    } \
    if (lane_id >= K_out && lane_id < k) { \
        outDistances[qi * k + lane_id] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + lane_id] = -1; \
    } \
}
SIMDGROUP_SELECT_VARIANT(32, 1)  /* k≤32: 1 per lane → 32 candidates */
SIMDGROUP_SELECT_VARIANT(64, 2)  /* k≤64: 2 per lane → 64 candidates */
#undef SIMDGROUP_SELECT_VARIANT

#define TOPK_THREADGROUP_VARIANT(K) \
kernel void topk_threadgroup_##K( \
    device const float* distances [[buffer(0)]], \
    device float* outDistances [[buffer(1)]], \
    device int* outIndices [[buffer(2)]], \
    device const uint* params [[buffer(3)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint TG_SIZE = 256; \
    constexpr uint R = 4;  /* local candidates per thread */ \
    constexpr uint MAX_CANDIDATES = TG_SIZE * R; \
    threadgroup float tgDist[MAX_CANDIDATES]; \
    threadgroup int tgIdx[MAX_CANDIDATES]; \
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    const device float* row = distances + qi * nb; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    \
    /* Per-thread: strided scan, keep local top-R in registers */ \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    \
    for (uint j = tid; j < nb; j += TG_SIZE) { \
        float v = row[j]; \
        /* Insert into local sorted list (keep best R) */ \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; \
                localIdx[pos] = localIdx[pos-1]; \
                pos--; \
            } \
            localDist[pos] = v; \
            localIdx[pos] = (int)j; \
            localCount++; \
        } else { \
            /* Check if better than worst */ \
            bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; \
                    localIdx[pos] = localIdx[pos-1]; \
                    pos--; \
                } \
                localDist[pos] = v; \
                localIdx[pos] = (int)j; \
            } \
        } \
    } \
    \
    /* Write local candidates to threadgroup memory */ \
    for (uint i = 0; i < R; i++) { \
        uint idx = tid * R + i; \
        if (i < localCount) { \
            tgDist[idx] = localDist[i]; \
            tgIdx[idx] = localIdx[i]; \
        } else { \
            tgDist[idx] = want_min ? 1e38f : -1e38f; \
            tgIdx[idx] = -1; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Reduction tree: compare-swap passes to get top K_out sorted */ \
    uint activeSize = min(MAX_CANDIDATES, nb); \
    /* Fixed number of passes: enough to get top K_out reasonably sorted */ \
    uint numPasses = min(activeSize, 32u);  /* Limit passes for performance */ \
    for (uint pass = 0; pass < numPasses; pass++) { \
        uint idx = tid; \
        if (idx >= activeSize - 1) { \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            continue; \
        } \
        /* Compare-swap pairs: even indices with next on even passes, odd on odd passes */ \
        bool doCompare = (pass % 2 == 0) ? (idx % 2 == 0) : (idx % 2 == 1); \
        if (doCompare && idx + 1 < activeSize) { \
            bool swap = want_min ? (tgDist[idx + 1] < tgDist[idx]) : (tgDist[idx + 1] > tgDist[idx]); \
            if (swap) { \
                float td = tgDist[idx]; tgDist[idx] = tgDist[idx + 1]; tgDist[idx + 1] = td; \
                int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[idx + 1]; tgIdx[idx + 1] = ti; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    /* Output: first K_out threads write results (sorted) */ \
    if (tid < K_out) { \
        outDistances[qi * k + tid] = tgDist[tid]; \
        outIndices[qi * k + tid] = tgIdx[tid]; \
    } \
    if (tid >= K_out && tid < k) { \
        outDistances[qi * k + tid] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + tid] = -1; \
    } \
}
TOPK_THREADGROUP_VARIANT(32)
TOPK_THREADGROUP_VARIANT(64)
TOPK_THREADGROUP_VARIANT(128)
TOPK_THREADGROUP_VARIANT(256)
TOPK_THREADGROUP_VARIANT(512)
TOPK_THREADGROUP_VARIANT(1024)
#undef TOPK_THREADGROUP_VARIANT

#define BITONIC_MERGE_TWO_SORTED_VARIANT(K) \
kernel void topk_merge_two_sorted_##K( \
    device const float* inA [[buffer(0)]], \
    device const int* inAIdx [[buffer(1)]], \
    device const float* inB [[buffer(2)]], \
    device const int* inBIdx [[buffer(3)]], \
    device float* outK [[buffer(4)]], \
    device int* outIdx [[buffer(5)]], \
    device const uint* params [[buffer(6)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint TG_SIZE = 256; \
    constexpr uint K_prime = K; \
    threadgroup float tgDist[K_prime * 2]; \
    threadgroup int tgIdx[K_prime * 2]; \
    uint nq = params[0], K_out = params[1], want_min = params[2]; \
    if (qi >= nq || K_out == 0) return; \
    uint K_actual = min(K_out, (uint)K_prime); \
    \
    /* Load both sorted lists into threadgroup memory */ \
    if (tid < K_actual) { \
        tgDist[tid] = inA[qi * K_out + tid]; \
        tgIdx[tid] = inAIdx[qi * K_out + tid]; \
        tgDist[K_actual + tid] = inB[qi * K_out + tid]; \
        tgIdx[K_actual + tid] = inBIdx[qi * K_out + tid]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Bitonic merge: merge two sorted lists of size K_actual into one sorted list */ \
    /* Use Batcher's odd-even merge network for 2K_actual → K_actual */ \
    uint totalSize = K_actual * 2; \
    /* Bitonic merge stages */ \
    for (uint stage = 1; stage < totalSize; stage *= 2) { \
        for (uint substage = stage; substage > 0; substage /= 2) { \
            uint idx = tid; \
            if (idx >= totalSize) { \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                continue; \
            } \
            uint partner = idx ^ substage; \
            if (partner < totalSize && partner > idx) { \
                bool swap = false; \
                if ((idx & stage) == 0) { \
                    /* Ascending merge */ \
                    swap = want_min ? (tgDist[partner] < tgDist[idx]) : (tgDist[partner] > tgDist[idx]); \
                } else { \
                    /* Descending merge */ \
                    swap = want_min ? (tgDist[idx] < tgDist[partner]) : (tgDist[idx] > tgDist[partner]); \
                } \
                if (swap) { \
                    float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td; \
                    int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti; \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    /* Output: first K_actual elements (best K′) */ \
    if (tid < K_actual) { \
        outK[qi * K_out + tid] = tgDist[tid]; \
        outIdx[qi * K_out + tid] = tgIdx[tid]; \
    } \
    if (tid >= K_actual && tid < K_out) { \
        outK[qi * K_out + tid] = want_min ? 1e38f : -1e38f; \
        outIdx[qi * K_out + tid] = -1; \
    } \
}
BITONIC_MERGE_TWO_SORTED_VARIANT(32)
BITONIC_MERGE_TWO_SORTED_VARIANT(64)
BITONIC_MERGE_TWO_SORTED_VARIANT(128)
BITONIC_MERGE_TWO_SORTED_VARIANT(256)
BITONIC_MERGE_TWO_SORTED_VARIANT(512)
BITONIC_MERGE_TWO_SORTED_VARIANT(1024)
#undef BITONIC_MERGE_TWO_SORTED_VARIANT

// Merge kernel: combines numTiles sorted lists (each of size k) into final top-k.
// LEGACY/FALLBACK: This heap-based merge is kept as fallback when bitonic merge unavailable.
// Prefer merge tree using topk_merge_two_sorted_K for better performance.
// Input: inK/inV are (nq, numTiles * k) - each row has numTiles chunks of k.
// Output: outK/outIdx are (nq, k) - final top-k per query.
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

kernel void trim_K_to_k(
    device const float* inK [[buffer(0)]],
    device const int* inIdx [[buffer(1)]],
    device float* outK [[buffer(2)]],
    device int* outIdx [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint qi [[thread_position_in_grid]]
) {
    uint nq = params[0], K_prime = params[1], k = params[2];
    if (qi >= nq || k == 0) return;
    uint kk = min(k, K_prime);
    for (uint i = 0; i < kk; i++) {
        outK[qi * k + i] = inK[qi * K_prime + i];
        outIdx[qi * k + i] = inIdx[qi * K_prime + i];
    }
    for (uint i = kk; i < k; i++) {
        outK[qi * k + i] = 1e38f;
        outIdx[qi * k + i] = -1;
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
// Simdgroup-based variants (for small k, tiled path)
static const char* kSimdgroupVariantNames[] = {
    "topk_simdgroup_32", "topk_simdgroup_64", nullptr, nullptr, nullptr, nullptr, nullptr
};
// Threadgroup-based parallel variants (for tiled path, medium/large k)
static const char* kThreadgroupVariantNames[] = {
    "topk_threadgroup_32", "topk_threadgroup_64", "topk_threadgroup_128", "topk_threadgroup_256",
    "topk_threadgroup_512", "topk_threadgroup_1024", nullptr  // 2048 not supported in threadgroup path
};
static const char* kMergeVariantNames[] = {
    "topk_merge_pair_32", "topk_merge_pair_64", "topk_merge_pair_128", "topk_merge_pair_256",
    "topk_merge_pair_512", "topk_merge_pair_1024", "topk_merge_pair_2048"
};
// Bitonic merge variants: merge two sorted lists of length K′ into one sorted list
static const char* kBitonicMergeVariantNames[] = {
    "topk_merge_two_sorted_32", "topk_merge_two_sorted_64", "topk_merge_two_sorted_128", "topk_merge_two_sorted_256",
    "topk_merge_two_sorted_512", "topk_merge_two_sorted_1024", nullptr  // 2048 not supported
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
    // K′ = 2k strategy: keep 2k candidates per tile and through merge, then trim to k
    // For k ≤ 256: K′ = 2k; for k > 256: K′ = 512 (generic path)
    int K_prime = needsTiling ? (k <= 256 ? (2 * k) : 512) : k;
    
    // Note: For k > 256, K′ = 512 (generic path), so threadgroupSelect is used.
    int variantIndexK = selectTopKVariant(K_prime);
    int variantIndex = selectTopKVariant(k);  // For old merge fallback, still use k
    bool useSimdgroup = needsTiling && K_prime <= 64 && kSimdgroupVariantNames[variantIndexK] != nullptr;
    bool useThreadgroup = needsTiling && !useSimdgroup && K_prime <= 1024;  // Threadgroup variants only go up to 1024
    
    id<MTLFunction> fnDist = [lib newFunctionWithName:isL2 ? @"l2_squared_matrix" : @"ip_matrix"];
    id<MTLFunction> fnTopK = nil;
    // Select top-k variant based on K′ (output size), not k
    if (useSimdgroup && kSimdgroupVariantNames[variantIndexK]) {
        fnTopK = [lib newFunctionWithName:@(kSimdgroupVariantNames[variantIndexK])];
    } else if (useThreadgroup && kThreadgroupVariantNames[variantIndexK]) {
        fnTopK = [lib newFunctionWithName:@(kThreadgroupVariantNames[variantIndexK])];
    } else {
        fnTopK = [lib newFunctionWithName:@(kTopKVariantNames[variantIndexK])];
    }
    //TODO: Remove old merge kernels
    // Use bitonic merge for merge tree (new approach), fallback to old heap merge if bitonic not available
    id<MTLFunction> fnBitonicMerge = nil;
    id<MTLFunction> fnMerge = nil;  // Old heap-based merge (kept for fallback or non-tree path)
    id<MTLFunction> fnIncrement = nil;
    id<MTLFunction> fnTrim = nil;
    if (needsTiling) {
        // Prefer bitonic merge for merge tree (use variant based on K′)
        int mergeVariantIndex = selectTopKVariant(K_prime);
        if (kBitonicMergeVariantNames[mergeVariantIndex]) {
            fnBitonicMerge = [lib newFunctionWithName:@(kBitonicMergeVariantNames[mergeVariantIndex])];
        }
        // Keep old merge as fallback (or for single-tile case)
        fnMerge = [lib newFunctionWithName:@(kMergeVariantNames[variantIndex])];
        fnIncrement = [lib newFunctionWithName:@"increment_index"];
        // Trim kernel: only needed when K′ > k
        if (K_prime > k) {
            fnTrim = [lib newFunctionWithName:@"trim_K_to_k"];
        }
    }
    
    if (!fnDist || !fnTopK || (needsTiling && (!fnIncrement || (!fnBitonicMerge && !fnMerge)))) {
        return false;
    }

    id<MTLComputePipelineState> psDist = [device newComputePipelineStateWithFunction:fnDist error:&err];
    id<MTLComputePipelineState> psTopK = [device newComputePipelineStateWithFunction:fnTopK error:&err];
    id<MTLComputePipelineState> psBitonicMerge = nil;
    id<MTLComputePipelineState> psMerge = nil;
    id<MTLComputePipelineState> psIncrement = nil;
    id<MTLComputePipelineState> psTrim = nil;
    if (needsTiling) {
        if (fnBitonicMerge) {
            psBitonicMerge = [device newComputePipelineStateWithFunction:fnBitonicMerge error:&err];
        }
        if (fnMerge) {
            psMerge = [device newComputePipelineStateWithFunction:fnMerge error:&err];
        }
        psIncrement = [device newComputePipelineStateWithFunction:fnIncrement error:&err];
        if (fnTrim) {
            psTrim = [device newComputePipelineStateWithFunction:fnTrim error:&err];
        }
        if (!psIncrement || (!psBitonicMerge && !psMerge) || (K_prime > k && !psTrim)) {
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
        
        // Temporary buffers for per-tile top-K′ results: (nq, numColTiles * K′)
        id<MTLBuffer> outDistanceBuf = [device newBufferWithLength:
            (size_t)nq * (size_t)numColTiles * (size_t)K_prime * sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIndexBuf = [device newBufferWithLength:
            (size_t)nq * (size_t)numColTiles * (size_t)K_prime * sizeof(int32_t)
            options:MTLResourceStorageModeShared];
        if (!outDistanceBuf || !outIndexBuf) {
            [enc endEncoding];
            return false;
        }
        
        // Tile over queries
        for (int tileRow = 0; tileRow < numRowTiles; tileRow++) {
            int curQuerySize = std::min(tileRows, nq - tileRow * tileRows);
            size_t queryBaseOffset = (size_t)(tileRow * tileRows) * (size_t)numColTiles * (size_t)K_prime;
            
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
                
                // Top-K′ for this tile: output K′ candidates (will be trimmed to k later)
                // Layout: [queryIdx][tileCol][K′] = queryIdx * (numColTiles * K′) + tileCol * K′
                [enc setComputePipelineState:psTopK];
                [enc setBuffer:distMatrixTile offset:0 atIndex:0];
                // Output offset for this tile: queryBaseOffset + queryIdx * (numColTiles * K′) + tileCol * K′
                // But top-k kernel expects contiguous output per query, so we need temp buffers
                id<MTLBuffer> tileOutDist = [device newBufferWithLength:
                    (size_t)curQuerySize * (size_t)K_prime * sizeof(float)
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> tileOutIdx = [device newBufferWithLength:
                    (size_t)curQuerySize * (size_t)K_prime * sizeof(int32_t)
                    options:MTLResourceStorageModeShared];
                if (!tileOutDist || !tileOutIdx) {
                    [enc endEncoding];
                    return false;
                }
                [enc setBuffer:tileOutDist offset:0 atIndex:1];
                [enc setBuffer:tileOutIdx offset:0 atIndex:2];
                uint32_t topkArgs[4] = {(uint32_t)curQuerySize, (uint32_t)curVecSize, (uint32_t)K_prime, isL2 ? 1u : 0u};
                [enc setBytes:topkArgs length:sizeof(topkArgs) atIndex:3];
                MTLSize gridTopK = MTLSizeMake(curQuerySize, 1, 1);
                // Use threadgroup size 32 for simdgroup, 256 for threadgroup, 1 for heap kernel
                MTLSize tgSizeTopK;
                if (useSimdgroup) {
                    tgSizeTopK = MTLSizeMake(32, 1, 1);  // One simdgroup per query
                } else if (useThreadgroup) {
                    tgSizeTopK = MTLSizeMake(256, 1, 1);  // Full threadgroup per query
                } else {
                    tgSizeTopK = MTLSizeMake(1, 1, 1);  // Single thread per query
                }
                [enc dispatchThreadgroups:gridTopK threadsPerThreadgroup:tgSizeTopK];
                
                // Copy tile results to main buffer (interleaved by tile column)
                // Use blit encoder for efficient copy
                [enc endEncoding];
                id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                for (int q = 0; q < curQuerySize; q++) {
                    size_t srcOffset = (size_t)q * (size_t)K_prime;
                    size_t dstOffset = queryBaseOffset + (size_t)q * (size_t)numColTiles * (size_t)K_prime + (size_t)tileCol * (size_t)K_prime;
                    [blitEnc copyFromBuffer:tileOutDist sourceOffset:srcOffset * sizeof(float)
                                 toBuffer:outDistanceBuf destinationOffset:dstOffset * sizeof(float)
                                 size:K_prime * sizeof(float)];
                    [blitEnc copyFromBuffer:tileOutIdx sourceOffset:srcOffset * sizeof(int32_t)
                                 toBuffer:outIndexBuf destinationOffset:dstOffset * sizeof(int32_t)
                                 size:K_prime * sizeof(int32_t)];
                }
                [blitEnc endEncoding];
                enc = [cmdBuf computeCommandEncoder];
            }
            
            // After all vector tiles for this query batch: adjust indices and merge
            // Adjust indices: add tileCols * tileCol to make them global (use K′)
            if (numColTiles > 1) {
                [enc setComputePipelineState:psIncrement];
                [enc setBuffer:outIndexBuf offset:queryBaseOffset * sizeof(int32_t) atIndex:0];
                uint32_t incArgs[4] = {(uint32_t)curQuerySize, (uint32_t)K_prime, (uint32_t)tileCols, (uint32_t)numColTiles};
                [enc setBytes:incArgs length:sizeof(incArgs) atIndex:1];
                // Grid: (numColTiles, curQuerySize) - one threadgroup per (tile, query) pair
                MTLSize gridInc = MTLSizeMake(numColTiles, curQuerySize, 1);
                [enc dispatchThreadgroups:gridInc threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
            
            // Merge per-tile top-k lists into final top-k using merge tree
            if (numColTiles > 1) {
                if (psBitonicMerge) {
                    // Merge tree: pairwise merge using bitonic merge kernel
                    // Process each query in the batch separately
                    // Double-buffer: ping-pong between outDistanceBuf/outIndexBuf and temp buffers
                    id<MTLBuffer> mergeBufA_Dist = outDistanceBuf;
                    id<MTLBuffer> mergeBufA_Idx = outIndexBuf;
                    id<MTLBuffer> mergeBufB_Dist = [device newBufferWithLength:
                        (size_t)curQuerySize * (size_t)numColTiles * (size_t)K_prime * sizeof(float)
                        options:MTLResourceStorageModeShared];
                    id<MTLBuffer> mergeBufB_Idx = [device newBufferWithLength:
                        (size_t)curQuerySize * (size_t)numColTiles * (size_t)K_prime * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
                    if (!mergeBufB_Dist || !mergeBufB_Idx) {
                        [enc endEncoding];
                        return false;
                    }
                    
                    // Merge tree: ceil(log2(numColTiles)) stages, per query
                    int numLists = numColTiles;
                    bool useBufA = true;
                    size_t stride = (size_t)numColTiles * (size_t)K_prime;  // Stride between queries
                    
                    while (numLists > 1) {
                        int numPairs = numLists / 2;
                        int numRemaining = numLists % 2;
                        
                        // Merge pairs for all queries in batch
                        for (int pair = 0; pair < numPairs; pair++) {
                            int listA_idx = pair * 2;
                            int listB_idx = pair * 2 + 1;
                            
                            id<MTLBuffer> srcA_Dist = useBufA ? mergeBufA_Dist : mergeBufB_Dist;
                            id<MTLBuffer> srcA_Idx = useBufA ? mergeBufA_Idx : mergeBufB_Idx;
                            id<MTLBuffer> srcB_Dist = useBufA ? mergeBufA_Dist : mergeBufB_Dist;
                            id<MTLBuffer> srcB_Idx = useBufA ? mergeBufA_Idx : mergeBufB_Idx;
                            id<MTLBuffer> dst_Dist = useBufA ? mergeBufB_Dist : mergeBufA_Dist;
                            id<MTLBuffer> dst_Idx = useBufA ? mergeBufB_Idx : mergeBufA_Idx;
                            
                            [enc setComputePipelineState:psBitonicMerge];
                            // For each query: offset = queryBaseOffset + q * (numColTiles * K′) + tileCol * K′
                            // We dispatch one threadgroup per query, so each handles its own offsets
                            size_t baseOffsetA = queryBaseOffset + (size_t)listA_idx * (size_t)K_prime;
                            size_t baseOffsetB = queryBaseOffset + (size_t)listB_idx * (size_t)K_prime;
                            size_t baseOffsetDst = queryBaseOffset + (size_t)pair * (size_t)K_prime;
                            
                            // Dispatch one threadgroup per query
                            for (int q = 0; q < curQuerySize; q++) {
                                size_t offsetA = baseOffsetA + (size_t)q * stride;
                                size_t offsetB = baseOffsetB + (size_t)q * stride;
                                size_t offsetDst = baseOffsetDst + (size_t)q * stride;
                                
                                [enc setBuffer:srcA_Dist offset:offsetA * sizeof(float) atIndex:0];
                                [enc setBuffer:srcA_Idx offset:offsetA * sizeof(int32_t) atIndex:1];
                                [enc setBuffer:srcB_Dist offset:offsetB * sizeof(float) atIndex:2];
                                [enc setBuffer:srcB_Idx offset:offsetB * sizeof(int32_t) atIndex:3];
                                [enc setBuffer:dst_Dist offset:offsetDst * sizeof(float) atIndex:4];
                                [enc setBuffer:dst_Idx offset:offsetDst * sizeof(int32_t) atIndex:5];
                                uint32_t mergeArgs[3] = {1u, (uint32_t)K_prime, isL2 ? 1u : 0u};  // nq=1 per dispatch, K′ size
                                [enc setBytes:mergeArgs length:sizeof(mergeArgs) atIndex:6];
                                MTLSize gridMerge = MTLSizeMake(1, 1, 1);  // One threadgroup per query
                                MTLSize tgSizeMerge = MTLSizeMake(256, 1, 1);
                                [enc dispatchThreadgroups:gridMerge threadsPerThreadgroup:tgSizeMerge];
                            }
                        }
                        
                        // Copy remaining list (if odd number) for all queries
                        if (numRemaining > 0) {
                            [enc endEncoding];
                            id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                            int remainingIdx = numPairs * 2;
                            id<MTLBuffer> src_Dist = useBufA ? mergeBufA_Dist : mergeBufB_Dist;
                            id<MTLBuffer> src_Idx = useBufA ? mergeBufA_Idx : mergeBufB_Idx;
                            id<MTLBuffer> dst_Dist = useBufA ? mergeBufB_Dist : mergeBufA_Dist;
                            id<MTLBuffer> dst_Idx = useBufA ? mergeBufB_Idx : mergeBufA_Idx;
                            for (int q = 0; q < curQuerySize; q++) {
                                size_t srcOffset = queryBaseOffset + (size_t)q * stride + (size_t)remainingIdx * (size_t)K_prime;
                                size_t dstOffset = queryBaseOffset + (size_t)q * stride + (size_t)numPairs * (size_t)K_prime;
                                [blitEnc copyFromBuffer:src_Dist sourceOffset:srcOffset * sizeof(float)
                                             toBuffer:dst_Dist destinationOffset:dstOffset * sizeof(float)
                                             size:K_prime * sizeof(float)];
                                [blitEnc copyFromBuffer:src_Idx sourceOffset:srcOffset * sizeof(int32_t)
                                             toBuffer:dst_Idx destinationOffset:dstOffset * sizeof(int32_t)
                                             size:K_prime * sizeof(int32_t)];
                            }
                            [blitEnc endEncoding];
                            enc = [cmdBuf computeCommandEncoder];
                        }
                        
                        numLists = numPairs + numRemaining;
                        useBufA = !useBufA;
                    }
                    
                    // Final result is in the current buffer - trim K′ to k if needed, or copy directly
                    id<MTLBuffer> final_Dist = useBufA ? mergeBufA_Dist : mergeBufB_Dist;
                    id<MTLBuffer> final_Idx = useBufA ? mergeBufA_Idx : mergeBufB_Idx;
                    if (K_prime > k && psTrim) {
                        // Trim K′ to k using trim kernel
                        [enc setComputePipelineState:psTrim];
                        for (int q = 0; q < curQuerySize; q++) {
                            size_t srcOffset = queryBaseOffset + (size_t)q * stride;
                            size_t dstOffset = (size_t)((tileRow * tileRows + q) * k);
                            [enc setBuffer:final_Dist offset:srcOffset * sizeof(float) atIndex:0];
                            [enc setBuffer:final_Idx offset:srcOffset * sizeof(int32_t) atIndex:1];
                            [enc setBuffer:outDistances offset:dstOffset * sizeof(float) atIndex:2];
                            [enc setBuffer:outIndices offset:dstOffset * sizeof(int32_t) atIndex:3];
                            uint32_t trimArgs[3] = {1u, (uint32_t)K_prime, (uint32_t)k};
                            [enc setBytes:trimArgs length:sizeof(trimArgs) atIndex:4];
                            MTLSize gridTrim = MTLSizeMake(1, 1, 1);
                            [enc dispatchThreadgroups:gridTrim threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        }
                    } else {
                        // Copy directly (K′ == k, no trim needed)
                        [enc endEncoding];
                        id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                        for (int q = 0; q < curQuerySize; q++) {
                            size_t finalSrcOffset = queryBaseOffset + (size_t)q * stride;
                            size_t finalDstOffset = (size_t)((tileRow * tileRows + q) * k);
                            [blitEnc copyFromBuffer:final_Dist sourceOffset:finalSrcOffset * sizeof(float)
                                         toBuffer:outDistances destinationOffset:finalDstOffset * sizeof(float)
                                         size:k * sizeof(float)];
                            [blitEnc copyFromBuffer:final_Idx sourceOffset:finalSrcOffset * sizeof(int32_t)
                                         toBuffer:outIndices destinationOffset:finalDstOffset * sizeof(int32_t)
                                         size:k * sizeof(int32_t)];
                        }
                        [blitEnc endEncoding];
                        enc = [cmdBuf computeCommandEncoder];
                    }
                } else {
                    // Fallback to old heap-based merge (uses k, not K′ - old kernel doesn't support K′)
                    // Create temp buffer for merge output, then trim if needed
                    id<MTLBuffer> mergeOutDist = [device newBufferWithLength:
                        (size_t)curQuerySize * (size_t)k * sizeof(float)
                        options:MTLResourceStorageModeShared];
                    id<MTLBuffer> mergeOutIdx = [device newBufferWithLength:
                        (size_t)curQuerySize * (size_t)k * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
                    if (!mergeOutDist || !mergeOutIdx) {
                        [enc endEncoding];
                        return false;
                    }
                    [enc setComputePipelineState:psMerge];
                    [enc setBuffer:outDistanceBuf offset:queryBaseOffset * sizeof(float) atIndex:0];
                    [enc setBuffer:outIndexBuf offset:queryBaseOffset * sizeof(int32_t) atIndex:1];
                    [enc setBuffer:mergeOutDist offset:0 atIndex:2];
                    [enc setBuffer:mergeOutIdx offset:0 atIndex:3];
                    uint32_t mergeArgs[4] = {(uint32_t)curQuerySize, (uint32_t)numColTiles, (uint32_t)k, isL2 ? 1u : 0u};
                    [enc setBytes:mergeArgs length:sizeof(mergeArgs) atIndex:4];
                    MTLSize gridMerge = MTLSizeMake(curQuerySize, 1, 1);
                    [enc dispatchThreadgroups:gridMerge threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    
                    // Copy merge output to final output
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                    size_t dstOffset = (size_t)(tileRow * tileRows * k);
                    [blitEnc copyFromBuffer:mergeOutDist sourceOffset:0
                                 toBuffer:outDistances destinationOffset:dstOffset * sizeof(float)
                                 size:curQuerySize * k * sizeof(float)];
                    [blitEnc copyFromBuffer:mergeOutIdx sourceOffset:0
                                 toBuffer:outIndices destinationOffset:dstOffset * sizeof(int32_t)
                                 size:curQuerySize * k * sizeof(int32_t)];
                    [blitEnc endEncoding];
                    enc = [cmdBuf computeCommandEncoder];
                }
            } else {
                // Single column tile: trim K′ to k if needed, or copy directly
                if (K_prime > k && psTrim) {
                    // Trim K′ to k
                    [enc setComputePipelineState:psTrim];
                    size_t srcOffset = (size_t)(tileRow * tileRows) * (size_t)K_prime;
                    size_t dstOffset = (size_t)(tileRow * tileRows * k);
                    [enc setBuffer:outDistanceBuf offset:srcOffset * sizeof(float) atIndex:0];
                    [enc setBuffer:outIndexBuf offset:srcOffset * sizeof(int32_t) atIndex:1];
                    [enc setBuffer:outDistances offset:dstOffset * sizeof(float) atIndex:2];
                    [enc setBuffer:outIndices offset:dstOffset * sizeof(int32_t) atIndex:3];
                    uint32_t trimArgs[3] = {(uint32_t)curQuerySize, (uint32_t)K_prime, (uint32_t)k};
                    [enc setBytes:trimArgs length:sizeof(trimArgs) atIndex:4];
                    MTLSize gridTrim = MTLSizeMake(curQuerySize, 1, 1);
                    [enc dispatchThreadgroups:gridTrim threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } else {
                    // Copy directly (K′ == k, no trim needed)
                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
                    size_t srcOffset = (size_t)(tileRow * tileRows) * (size_t)K_prime;
                    size_t dstOffset = (size_t)(tileRow * tileRows * k);
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
