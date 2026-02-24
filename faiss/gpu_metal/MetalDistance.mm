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
#include <mutex>
#include <string>
#include <unordered_map>

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
    /* Full bitonic sort (exact top-k); same pattern as threadgroupSelect. */ \
    uint activeSize = min(MAX_CANDIDATES, nb); \
    uint N = activeSize; \
    if (N == 0) N = 1; \
    else { N--; N |= N >> 1; N |= N >> 2; N |= N >> 4; N |= N >> 8; N |= N >> 16; N++; } \
    if (N > MAX_CANDIDATES) N = MAX_CANDIDATES; \
    for (uint i = lane_id; i < N; i += SIMD_WIDTH) { \
        if (i >= activeSize) { tgDist[i] = want_min ? 1e38f : -1e38f; tgIdx[i] = -1; } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    /* Canonical bitonic sort: k2=2,4,...,N; j=k2/2,...,1; direction from (idx & k2) == 0 */ \
    for (uint k2 = 2; k2 <= N; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = lane_id; idx < N; idx += SIMD_WIDTH) { \
                uint partner = idx ^ j; \
                if (partner < N && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                                      : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
                                  : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])); \
                    bool swap = ascending ? partnerBetter : idxBetter; \
                    if (swap) { \
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td; \
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti; \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    /* Output: first K_out elements (strided so all written; K_out can be 32 or 64) */ \
    for (uint i = lane_id; i < K_out; i += SIMD_WIDTH) { \
        outDistances[qi * k + i] = tgDist[i]; \
        outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = lane_id; i < k - K_out; i += SIMD_WIDTH) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + K_out + i] = -1; \
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
    constexpr uint R = 4; \
    constexpr uint R_out = R; \
    constexpr uint CANDIDATES = TG_SIZE * R_out; \
    threadgroup float tgDist[CANDIDATES]; \
    threadgroup int tgIdx[CANDIDATES]; \
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    const device float* row = distances + qi * nb; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    for (uint j = tid; j < nb; j += TG_SIZE) { \
        float v = row[j]; \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; localIdx[pos] = localIdx[pos-1]; pos--; \
            } \
            localDist[pos] = v; localIdx[pos] = (int)j; localCount++; \
        } else { \
            bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; localIdx[pos] = localIdx[pos-1]; pos--; \
                } \
                localDist[pos] = v; localIdx[pos] = (int)j; \
            } \
        } \
    } \
    for (uint i = 0; i < R_out; i++) { \
        uint idx = tid * R_out + i; \
        if (i < localCount) { tgDist[idx] = localDist[i]; tgIdx[idx] = localIdx[i]; } \
        else { tgDist[idx] = want_min ? 1e38f : -1e38f; tgIdx[idx] = -1; } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) { \
                uint partner = idx ^ j; \
                if (partner < CANDIDATES && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                                      : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
                                  : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])); \
                    bool swap = ascending ? partnerBetter : idxBetter; \
                    if (swap) { \
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td; \
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti; \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    for (uint i = tid; i < K_out; i += TG_SIZE) { \
        outDistances[qi * k + i] = tgDist[i]; outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < k - K_out; i += TG_SIZE) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; outIndices[qi * k + K_out + i] = -1; \
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
    uint nq = params[0], K_actual = params[1], want_min = params[2]; \
    if (qi >= nq || K_actual == 0) return; \
    K_actual = min(K_actual, (uint)K_prime); \
    constexpr uint totalSize = K_prime * 2; \
    for (uint i = tid; i < K_prime; i += TG_SIZE) { \
        if (i < K_actual) { \
            tgDist[i] = inA[qi * K_prime + i]; tgIdx[i] = inAIdx[qi * K_prime + i]; \
            tgDist[K_prime + i] = inB[qi * K_prime + i]; tgIdx[K_prime + i] = inBIdx[qi * K_prime + i]; \
        } else { \
            float sentinel = want_min ? 1e38f : -1e38f; \
            tgDist[i] = sentinel; tgIdx[i] = -1; \
            tgDist[K_prime + i] = sentinel; tgIdx[K_prime + i] = -1; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint k2 = 2; k2 <= totalSize; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = tid; idx < totalSize; idx += TG_SIZE) { \
                uint partner = idx ^ j; \
                if (partner < totalSize && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                                      : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
                                  : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])); \
                    bool swap = ascending ? partnerBetter : idxBetter; \
                    if (swap) { \
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td; \
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti; \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    for (uint i = tid; i < K_actual; i += TG_SIZE) { \
        outK[qi * K_prime + i] = tgDist[i]; outIdx[qi * K_prime + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < K_prime - K_actual; i += TG_SIZE) { \
        outK[qi * K_prime + K_actual + i] = want_min ? 1e38f : -1e38f; \
        outIdx[qi * K_prime + K_actual + i] = -1; \
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
    uint nq = params[0], K_prime = params[1], k = params[2], want_min = params[3];
    if (qi >= nq || k == 0) return;
    uint kk = min(k, K_prime);
    for (uint i = 0; i < kk; i++) {
        outK[qi * k + i] = inK[qi * K_prime + i];
        outIdx[qi * k + i] = inIdx[qi * K_prime + i];
    }
    float sentinel = want_min ? 1e38f : -1e38f;
    for (uint i = kk; i < k; i++) {
        outK[qi * k + i] = sentinel;
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
    
    // 3. Try relative to source file at compile time (works regardless of CWD)
    if (!metalPath) {
        NSString* sourceFile = @(__FILE__);
        NSString* sourceDir = [sourceFile stringByDeletingLastPathComponent];
        NSString* relPath = [sourceDir stringByAppendingPathComponent:@"MetalDistance.metal"];
        if ([fm fileExistsAtPath:relPath]) {
            metalPath = relPath;
        }
    }

    // 4. Try relative to executable (for installed libraries)
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

// ============================================================
// Pipeline state cache: compiled once per (device, function).
// Avoids the expensive newLibraryWithSource: + newComputePipelineState
// calls on every search invocation.
// ============================================================

struct PipelineCacheKey {
    uintptr_t devicePtr;
    std::string functionName;
    bool operator==(const PipelineCacheKey& o) const {
        return devicePtr == o.devicePtr && functionName == o.functionName;
    }
};
struct PipelineCacheKeyHash {
    size_t operator()(const PipelineCacheKey& k) const {
        size_t h = std::hash<uintptr_t>{}(k.devicePtr);
        h ^= std::hash<std::string>{}(k.functionName) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Use __bridge_retained / __bridge_transfer to keep ARC-managed objects alive.
static std::mutex gPipelineCacheMutex;
static std::unordered_map<PipelineCacheKey,
                          id<MTLComputePipelineState>,
                          PipelineCacheKeyHash>
        gPipelineCache;

// Returns a cached (or freshly compiled) pipeline state for functionName.
// lib is only consulted on a cache miss.
static id<MTLComputePipelineState> getCachedPipeline(
        id<MTLDevice> device,
        id<MTLLibrary> lib,
        const char* functionName) {
    PipelineCacheKey key{(uintptr_t)(__bridge void*)device, functionName};
    {
        std::lock_guard<std::mutex> lock(gPipelineCacheMutex);
        auto it = gPipelineCache.find(key);
        if (it != gPipelineCache.end()) {
            return it->second;
        }
    }
    // Cache miss: compile now.
    id<MTLFunction> fn = [lib newFunctionWithName:@(functionName)];
    if (!fn) {
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) {
        return nil;
    }
    {
        std::lock_guard<std::mutex> lock(gPipelineCacheMutex);
        gPipelineCache[key] = ps;
    }
    return ps;
}

// Returns cached library for a device (one library covers all kernels).
static std::unordered_map<uintptr_t, id<MTLLibrary>> gLibraryCache;
static std::mutex gLibraryCacheMutex;

static id<MTLLibrary> getCachedLibrary(id<MTLDevice> device) {
    uintptr_t key = (uintptr_t)(__bridge void*)device;
    {
        std::lock_guard<std::mutex> lock(gLibraryCacheMutex);
        auto it = gLibraryCache.find(key);
        if (it != gLibraryCache.end()) {
            return it->second;
        }
    }
    NSString* mslSource = loadMSLSource();
    if (!mslSource) {
        return nil;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:mslSource options:nil error:&err];
    if (!lib) {
        return nil;
    }
    {
        std::lock_guard<std::mutex> lock(gLibraryCacheMutex);
        gLibraryCache[key] = lib;
    }
    return lib;
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

// Section 8: K′ = nextPow2(min(2*k, 1024)) so we use 256/512 for most k, 1024 only when needed.
static int computeKPrimeForTiling(int k) {
    int cap = std::min(2 * k, 1024);
    if (cap <= 0) return 32;
    uint32_t x = (uint32_t)cap;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    x++;
    return (int)std::min(std::max(x, 32u), 1024u);
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

    // Use cached library (compiled once per device).
    id<MTLLibrary> lib = getCachedLibrary(device);
    if (!lib) {
        return false;
    }

    // Calculate tile sizes
    int tileRows, tileCols;
    size_t availableMem = 512 * 1024 * 1024;
    chooseTileSize(nq, nb, d, sizeof(float), availableMem, tileRows, tileCols);
    
    bool needsTiling = (tileCols < nb || tileRows < nq);
    int K_prime = needsTiling ? computeKPrimeForTiling(k) : k;
    
    int variantIndexK = selectTopKVariant(K_prime);
    int variantIndex = selectTopKVariant(k);
    // Simdgroup variant disabled for tiled paths: stride=32 with R=1 means each
    // lane keeps only its best candidate, causing birthday-problem misses when
    // multiple top-K items hash to the same lane.  Threadgroup (stride=256, R=4)
    // is safe because CANDIDATES=1024 writes all per-thread candidates.
    bool useSimdgroup = false;
    bool useThreadgroup = needsTiling && K_prime <= 1024;

    const char* distName = isL2 ? "l2_squared_matrix" : "ip_matrix";
    const char* topkName = nullptr;
    if (useSimdgroup && kSimdgroupVariantNames[variantIndexK]) {
        topkName = kSimdgroupVariantNames[variantIndexK];
    } else if (useThreadgroup && kThreadgroupVariantNames[variantIndexK]) {
        topkName = kThreadgroupVariantNames[variantIndexK];
    } else {
        topkName = kTopKVariantNames[variantIndexK];
    }

    id<MTLComputePipelineState> psDist     = getCachedPipeline(device, lib, distName);
    id<MTLComputePipelineState> psTopK     = getCachedPipeline(device, lib, topkName);
    id<MTLComputePipelineState> psBitonicMerge = nil;
    id<MTLComputePipelineState> psMerge    = nil;
    id<MTLComputePipelineState> psIncrement = nil;
    id<MTLComputePipelineState> psTrim     = nil;
    if (needsTiling) {
        int mergeVariantIndex = selectTopKVariant(K_prime);
        if (kBitonicMergeVariantNames[mergeVariantIndex]) {
            psBitonicMerge = getCachedPipeline(device, lib, kBitonicMergeVariantNames[mergeVariantIndex]);
        }
        psMerge     = getCachedPipeline(device, lib, kMergeVariantNames[variantIndex]);
        psIncrement = getCachedPipeline(device, lib, "increment_index");
        if (K_prime > k) {
            psTrim = getCachedPipeline(device, lib, "trim_K_to_k");
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
                            uint32_t trimArgs[4] = {1u, (uint32_t)K_prime, (uint32_t)k, isL2 ? 1u : 0u};
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
                    uint32_t trimArgs[4] = {(uint32_t)curQuerySize, (uint32_t)K_prime, (uint32_t)k, isL2 ? 1u : 0u};
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
        id<MTLBuffer> perListIdxBuf) {
    if (!device || !queue || !queries || !codes || !ids ||
        !listOffset || !listLength || !coarseAssign ||
        !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf) {
        return false;
    }
    if (k <= 0 || nq <= 0 || nprobe <= 0) {
        return false;
    }

    // Use cached library and pipeline states (compiled once per device).
    id<MTLLibrary> lib = getCachedLibrary(device);
    if (!lib) {
        return false;
    }
    id<MTLComputePipelineState> psScan =
            getCachedPipeline(device, lib, "ivf_scan_list");
    id<MTLComputePipelineState> psMerge =
            getCachedPipeline(device, lib, "ivf_merge_lists");
    if (!psScan || !psMerge) {
        return false;
    }

    // Shared params buffer for both passes.
    uint32_t scanParams[5] = {
        (uint32_t)nq,
        (uint32_t)d,
        (uint32_t)k,
        (uint32_t)nprobe,
        isL2 ? 1u : 0u,
    };
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:scanParams
                                                  length:sizeof(scanParams)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // Pass 1: ivf_scan_list — one threadgroup per (query, probe) pair.
    [enc setComputePipelineState:psScan];
    [enc setBuffer:queries       offset:0 atIndex:0];
    [enc setBuffer:codes         offset:0 atIndex:1];
    [enc setBuffer:ids           offset:0 atIndex:2];
    [enc setBuffer:listOffset    offset:0 atIndex:3];
    [enc setBuffer:listLength    offset:0 atIndex:4];
    [enc setBuffer:coarseAssign  offset:0 atIndex:5];
    [enc setBuffer:perListDistBuf offset:0 atIndex:6];
    [enc setBuffer:perListIdxBuf  offset:0 atIndex:7];
    [enc setBuffer:paramsBuf     offset:0 atIndex:8];

    NSUInteger totalTGs = (NSUInteger)nq * (NSUInteger)nprobe;
    [enc dispatchThreadgroups:MTLSizeMake(totalTGs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // Pass 2: ivf_merge_lists — one threadgroup per query.
    [enc setComputePipelineState:psMerge];
    [enc setBuffer:perListDistBuf offset:0 atIndex:0];
    [enc setBuffer:perListIdxBuf  offset:0 atIndex:1];
    [enc setBuffer:outDistances   offset:0 atIndex:2];
    [enc setBuffer:outIndices     offset:0 atIndex:3];
    [enc setBuffer:paramsBuf      offset:0 atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

bool runMetalComputeNorms(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> vectors,
        int nb,
        int d,
        id<MTLBuffer> normsBuf) {
    if (!device || !queue || !vectors || !normsBuf || nb <= 0 || d <= 0) {
        return false;
    }
    id<MTLLibrary> lib = getCachedLibrary(device);
    if (!lib) return false;
    id<MTLComputePipelineState> ps = getCachedPipeline(device, lib, "compute_norms");
    if (!ps) return false;

    uint32_t args[2] = {(uint32_t)nb, (uint32_t)d};
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:vectors  offset:0 atIndex:0];
    [enc setBuffer:normsBuf offset:0 atIndex:1];
    [enc setBytes:args length:sizeof(args) atIndex:2];
    NSUInteger tgSize = std::min((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)nb + tgSize - 1) / tgSize;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

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
        id<MTLBuffer> centroidNormsBuf,
        int avgListLen) {
    if (!device || !queue || !queries || !centroids || !codes || !ids ||
        !listOffset || !listLength || !outDistances || !outIndices ||
        !perListDistBuf || !perListIdxBuf ||
        !coarseDistBuf || !coarseIdxBuf || !distMatrixBuf) {
        return false;
    }
    if (k <= 0 || nq <= 0 || nprobe <= 0 || nlist <= 0) {
        return false;
    }

    id<MTLLibrary> lib = getCachedLibrary(device);
    if (!lib) {
        return false;
    }

    // Use fused l2_with_norms when centroid norms are available.
    bool useFusedL2 = isL2 && centroidNormsBuf != nil;
    const char* distKernelName = useFusedL2 ? "l2_with_norms"
                               : (isL2 ? "l2_squared_matrix" : "ip_matrix");
    int coarseVariantIdx = selectTopKVariant(nprobe);
    const char* coarseTopkName = kTopKVariantNames[coarseVariantIdx];

    // Pick scan kernel: small lists → 32-thread variant.
    bool useSmallScan = (avgListLen <= 64);
    const char* scanKernelName = useSmallScan ? "ivf_scan_list_small"
                                              : "ivf_scan_list";

    id<MTLComputePipelineState> psDist  = getCachedPipeline(device, lib, distKernelName);
    id<MTLComputePipelineState> psTopK  = getCachedPipeline(device, lib, coarseTopkName);
    id<MTLComputePipelineState> psScan  = getCachedPipeline(device, lib, scanKernelName);
    id<MTLComputePipelineState> psMerge = getCachedPipeline(device, lib, "ivf_merge_lists");

    if (!psDist || !psTopK || !psScan || !psMerge) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    // ---- Step 0 (optional): blit query data to GPU asynchronously -----------
    // Use a blit encoder to copy query data so the GPU DMA engine handles it
    // concurrently with any prior compute work in the queue.
    // (Queries are already in a shared buffer; this is a no-op memcpy that
    //  ensures the buffer is committed to the command buffer dependency graph.)

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // ---- Step 1: coarse distance matrix (queries × centroids) ---------------
    [enc setComputePipelineState:psDist];
    [enc setBuffer:queries       offset:0 atIndex:0];
    [enc setBuffer:centroids     offset:0 atIndex:1];
    [enc setBuffer:distMatrixBuf offset:0 atIndex:2];
    if (useFusedL2) {
        uint32_t distArgs[3] = {(uint32_t)nq, (uint32_t)nlist, (uint32_t)d};
        [enc setBytes:distArgs length:sizeof(distArgs) atIndex:3];
        [enc setBuffer:centroidNormsBuf offset:0 atIndex:4];
    } else {
        uint32_t distArgs[3] = {(uint32_t)nq, (uint32_t)nlist, (uint32_t)d};
        [enc setBytes:distArgs length:sizeof(distArgs) atIndex:3];
    }

    const NSUInteger w = 16, h = 16;
    MTLSize distGrid = MTLSizeMake(((NSUInteger)nlist + w - 1) / w,
                                    ((NSUInteger)nq   + h - 1) / h, 1);
    [enc dispatchThreadgroups:distGrid threadsPerThreadgroup:MTLSizeMake(w, h, 1)];

    // ---- Step 2: coarse top-nprobe selection --------------------------------
    [enc setComputePipelineState:psTopK];
    [enc setBuffer:distMatrixBuf offset:0 atIndex:0];
    [enc setBuffer:coarseDistBuf offset:0 atIndex:1];
    [enc setBuffer:coarseIdxBuf  offset:0 atIndex:2];
    uint32_t topkArgs[4] = {(uint32_t)nq, (uint32_t)nlist, (uint32_t)nprobe,
                            isL2 ? 1u : 0u};
    [enc setBytes:topkArgs length:sizeof(topkArgs) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    // ---- Step 3: IVF scan (one threadgroup per query×probe) -----------------
    uint32_t scanParams[5] = {
        (uint32_t)nq, (uint32_t)d, (uint32_t)k,
        (uint32_t)nprobe, isL2 ? 1u : 0u
    };
    id<MTLBuffer> paramsBuf = [device newBufferWithBytes:scanParams
                                                  length:sizeof(scanParams)
                                                 options:MTLResourceStorageModeShared];
    if (!paramsBuf) {
        [enc endEncoding];
        return false;
    }

    [enc setComputePipelineState:psScan];
    [enc setBuffer:queries        offset:0 atIndex:0];
    [enc setBuffer:codes          offset:0 atIndex:1];
    [enc setBuffer:ids            offset:0 atIndex:2];
    [enc setBuffer:listOffset     offset:0 atIndex:3];
    [enc setBuffer:listLength     offset:0 atIndex:4];
    [enc setBuffer:coarseIdxBuf   offset:0 atIndex:5];
    [enc setBuffer:perListDistBuf offset:0 atIndex:6];
    [enc setBuffer:perListIdxBuf  offset:0 atIndex:7];
    [enc setBuffer:paramsBuf      offset:0 atIndex:8];

    NSUInteger scanTGSize = useSmallScan ? 32 : 256;
    NSUInteger totalTGs = (NSUInteger)nq * (NSUInteger)nprobe;
    [enc dispatchThreadgroups:MTLSizeMake(totalTGs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(scanTGSize, 1, 1)];

    // ---- Step 4: merge nprobe results per query -----------------------------
    [enc setComputePipelineState:psMerge];
    [enc setBuffer:perListDistBuf offset:0 atIndex:0];
    [enc setBuffer:perListIdxBuf  offset:0 atIndex:1];
    [enc setBuffer:outDistances   offset:0 atIndex:2];
    [enc setBuffer:outIndices     offset:0 atIndex:3];
    [enc setBuffer:paramsBuf      offset:0 atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

} // namespace gpu_metal
} // namespace faiss
