// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal Shading Language kernels for distance computation and top-k selection.
 * Used by MetalDistance.mm for reusable distance computation.
 *
 * Kernel organization:
 * - Distance kernels: l2_squared_matrix, ip_matrix (compute distance matrices)
 * - Per-tile selection (parallel):
 *   - simdgroupSelect: topk_simdgroup_32/64 (K′≤64, minimal threadgroup)
 *   - threadgroupSelect: topk_threadgroup_K (K′≤1024, parallel reduction)
 *   - Legacy heap: topk_heap_K (non-tiled path or fallback)
 * - Merge (pairwise merge tree):
 *   - Bitonic merge: topk_merge_two_sorted_K (preferred, parallel)
 *   - Legacy heap merge: topk_merge_pair_K (fallback, serial)
 * - Utilities: increment_index (adjust tile indices), trim_K_to_k (trim 2k→k)
 */

#include <metal_stdlib>
using namespace metal;

kernel void l2_squared_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // nq, nb, d
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
    device const uint* params [[buffer(3)]],  // nq, nb, d
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

//TODO: Remove old merge kernels/Legacey kernels
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

// Simdgroup-based top-k (warpSelect analogue): one simdgroup per query.
// Each lane keeps best 1 (or 2) in registers, then bitonic merge across lanes.
// Minimal threadgroup memory; good for small k (≤32, optionally ≤64).
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

// Parallel threadgroup-based top-k: one threadgroup per query, 256 threads per threadgroup
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
    constexpr uint R = 4;  /* local candidates per thread during scan */ \
    constexpr uint R_out = (K + 255) / 256;  /* 1 for K≤256, 2 for 512, 4 for 1024 */ \
    constexpr uint CANDIDATES = TG_SIZE * R_out;  /* 256, 256, 256, 256, 512, 1024 */ \
    threadgroup float tgDist[CANDIDATES]; \
    threadgroup int tgIdx[CANDIDATES]; \
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
    /* Write only best R_out per thread → exactly CANDIDATES total (Section 8) */ \
    for (uint i = 0; i < R_out; i++) { \
        uint idx = tid * R_out + i; \
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
    /* Sort exactly N = CANDIDATES (already power of two) */ \
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
    \
    /* Output: first K_out elements */ \
    for (uint i = tid; i < K_out; i += TG_SIZE) { \
        outDistances[qi * k + i] = tgDist[i]; \
        outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < k - K_out; i += TG_SIZE) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + K_out + i] = -1; \
    } \
}
TOPK_THREADGROUP_VARIANT(32)
TOPK_THREADGROUP_VARIANT(64)
TOPK_THREADGROUP_VARIANT(128)
TOPK_THREADGROUP_VARIANT(256)
TOPK_THREADGROUP_VARIANT(512)
TOPK_THREADGROUP_VARIANT(1024)
#undef TOPK_THREADGROUP_VARIANT

// Bitonic merge kernel: merges two sorted lists of length K′ into one sorted list of length K′.
// Input: two buffers A and B, each (nq, K′) sorted (ascending for L2, descending for IP).
// Output: one buffer C (nq, K′) sorted, containing best K′ from A and B combined.
// Uses one threadgroup per query with bitonic merge network (compare-exchange pattern).
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
    /* Params: nq, K_actual (valid entries), want_min. Stride is always K_prime (buffer layout). */ \
    uint nq = params[0], K_actual = params[1], want_min = params[2]; \
    if (qi >= nq || K_actual == 0) return; \
    K_actual = min(K_actual, (uint)K_prime); \
    \
    /* Load both lists; stride = K_prime. Pad to 2*K_prime so bitonic runs over power-of-two (Landmine B). */ \
    constexpr uint totalSize = K_prime * 2; \
    for (uint i = tid; i < K_prime; i += TG_SIZE) { \
        if (i < K_actual) { \
            tgDist[i] = inA[qi * K_prime + i]; \
            tgIdx[i] = inAIdx[qi * K_prime + i]; \
            tgDist[K_prime + i] = inB[qi * K_prime + i]; \
            tgIdx[K_prime + i] = inBIdx[qi * K_prime + i]; \
        } else { \
            float sentinel = want_min ? 1e38f : -1e38f; \
            tgDist[i] = sentinel; tgIdx[i] = -1; \
            tgDist[K_prime + i] = sentinel; tgIdx[K_prime + i] = -1; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Canonical bitonic merge: k2=2,4,...,totalSize; j=k2/2,...,1; direction from (idx & k2) == 0 */ \
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
    \
    /* Output: first K_actual elements; stride = K_prime. Fill remainder of row with sentinels if needed. */ \
    for (uint i = tid; i < K_actual; i += TG_SIZE) { \
        outK[qi * K_prime + i] = tgDist[i]; \
        outIdx[qi * K_prime + i] = tgIdx[i]; \
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
//TODO: Remove old merge kernels/Legacey kernels
// DEPRECATED: This heap-based merge will be replaced by merge tree using topk_merge_two_sorted_K.
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

// Increment indices: for each chunk of k indices, add tileCols * tileIndex.
// Used to convert tile-relative indices (0..tileCols-1) to global indices.
// Grid: (numTiles, nq) - one thread per (tile, query) pair processes k indices.
kernel void increment_index(
    device int* indices [[buffer(0)]],
    device const uint* params [[buffer(1)]],  // nq, k, tileCols, numTiles
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

// Trim kernel: copies first k elements from sorted K′ to output k. Generic for L2 (want_min) and IP (want_max).
// Input: inK/inIdx are (nq, K′) sorted.
// Output: outK/outIdx are (nq, k) - first k elements.
kernel void trim_K_to_k(
    device const float* inK [[buffer(0)]],
    device const int* inIdx [[buffer(1)]],
    device float* outK [[buffer(2)]],
    device int* outIdx [[buffer(3)]],
    device const uint* params [[buffer(4)]],  // nq, K_prime, k, want_min
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
