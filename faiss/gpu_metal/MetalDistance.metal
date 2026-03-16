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
 * - Distance kernels: l2_squared_matrix, ip_matrix (tiled GEMM-style), l2_with_norms, compute_norms
 * - Fused distance+top-k: fused_dist_topk_K (single-pass, no intermediate matrix)
 * - Top-k selection (parallel):
 *   - simdgroupSelect: topk_simdgroup_32/64 (K′≤64, minimal threadgroup)
 *   - threadgroupSelect: topk_threadgroup_K (K′≤2048, parallel reduction)
 * - Merge: topk_merge_two_sorted_K (bitonic merge, parallel)
 * - Utilities: increment_index (adjust tile indices), trim_K_to_k (trim 2k→k)
 * - IVF: ivf_scan_list, ivf_scan_list_small, ivf_scan_list_interleaved, ivf_merge_lists
 */

#include <metal_stdlib>
using namespace metal;

kernel void l2_squared_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);

        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            float d00 = q0 - v0; acc00 += d00 * d00;
            float d01 = q0 - v1; acc01 += d01 * d01;
            float d10 = q1 - v0; acc10 += d10 * d10;
            float d11 = q1 - v1; acc11 += d11 * d11;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

kernel void ip_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);

        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            acc00 += q0 * v0; acc01 += q0 * v1;
            acc10 += q1 * v0; acc11 += q1 * v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

// ============================================================
//  Float16 vector variants: queries are float32, vectors are half
// ============================================================

kernel void l2_squared_matrix_fp16(
    device const float* queries [[buffer(0)]],
    device const half*  vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);
        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? float(vectors[gCol * d + dk + mk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            float d00 = q0 - v0; acc00 += d00 * d00;
            float d01 = q0 - v1; acc01 += d01 * d01;
            float d10 = q1 - v0; acc10 += d10 * d10;
            float d11 = q1 - v1; acc11 += d11 * d11;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

kernel void ip_matrix_fp16(
    device const float* queries [[buffer(0)]],
    device const half*  vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);
        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? float(vectors[gCol * d + dk + mk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            acc00 += q0 * v0; acc01 += q0 * v1;
            acc10 += q1 * v0; acc11 += q1 * v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

#define FUSED_DIST_TOPK_FP16_VARIANT(K) \
kernel void fused_dist_topk_fp16_##K( \
    device const float* queries [[buffer(0)]], \
    device const half*  vectors [[buffer(1)]], \
    device float* outDistances [[buffer(2)]], \
    device int* outIndices [[buffer(3)]], \
    device const uint* params [[buffer(4)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint TG_SIZE = 256; \
    constexpr uint R = 4; \
    constexpr uint CANDIDATES = TG_SIZE * R; \
    constexpr uint MAX_DIM = 2048; \
    \
    threadgroup float tgQ[MAX_DIM]; \
    threadgroup float tgDist[CANDIDATES]; \
    threadgroup int tgIdx[CANDIDATES]; \
    \
    uint nq = params[0], nb = params[1], d = params[2]; \
    uint k = params[3], metric = params[4]; \
    bool want_min = (metric == 0); \
    if (qi >= nq || k == 0) return; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    \
    for (uint i = tid; i < d; i += TG_SIZE) \
        tgQ[i] = queries[qi * d + i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    \
    for (uint j = tid; j < nb; j += TG_SIZE) { \
        const device half* vec = vectors + j * d; \
        float dist = 0.0f; \
        if (metric == 0) { \
            for (uint dd = 0; dd < d; dd++) { \
                float diff = tgQ[dd] - float(vec[dd]); \
                dist += diff * diff; \
            } \
        } else { \
            for (uint dd = 0; dd < d; dd++) \
                dist += tgQ[dd] * float(vec[dd]); \
        } \
        \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && dist < localDist[pos-1]) || (!want_min && dist > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; \
                localIdx[pos] = localIdx[pos-1]; \
                pos--; \
            } \
            localDist[pos] = dist; \
            localIdx[pos] = (int)j; \
            localCount++; \
        } else { \
            bool better = want_min ? (dist < localDist[R-1]) : (dist > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && dist < localDist[pos-1]) || (!want_min && dist > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; \
                    localIdx[pos] = localIdx[pos-1]; \
                    pos--; \
                } \
                localDist[pos] = dist; \
                localIdx[pos] = (int)j; \
            } \
        } \
    } \
    \
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
    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) { \
                uint partner = idx ^ j; \
                if (partner < CANDIDATES && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min \
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min \
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
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
    for (uint i = tid; i < K_out; i += TG_SIZE) { \
        outDistances[qi * k + i] = tgDist[i]; \
        outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < k - K_out; i += TG_SIZE) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + K_out + i] = -1; \
    } \
}
FUSED_DIST_TOPK_FP16_VARIANT(32)
FUSED_DIST_TOPK_FP16_VARIANT(64)
FUSED_DIST_TOPK_FP16_VARIANT(128)
FUSED_DIST_TOPK_FP16_VARIANT(256)
FUSED_DIST_TOPK_FP16_VARIANT(512)
FUSED_DIST_TOPK_FP16_VARIANT(1024)
#undef FUSED_DIST_TOPK_FP16_VARIANT

// Fused distance + top-k: one threadgroup per query, compute distances on the fly
// and feed directly into local top-R buffers.  Eliminates the intermediate distance
// matrix for the non-tiled path.  Query is cached in threadgroup memory.
// params: [nq, nb, d, k, metric]  metric: 0 = L2, 1 = IP
#define FUSED_DIST_TOPK_VARIANT(K) \
kernel void fused_dist_topk_##K( \
    device const float* queries [[buffer(0)]], \
    device const float* vectors [[buffer(1)]], \
    device float* outDistances [[buffer(2)]], \
    device int* outIndices [[buffer(3)]], \
    device const uint* params [[buffer(4)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint TG_SIZE = 256; \
    constexpr uint R = 4; \
    constexpr uint CANDIDATES = TG_SIZE * R; /* 1024 */ \
    constexpr uint MAX_DIM = 2048; \
    \
    threadgroup float tgQ[MAX_DIM]; \
    threadgroup float tgDist[CANDIDATES]; \
    threadgroup int tgIdx[CANDIDATES]; \
    \
    uint nq = params[0], nb = params[1], d = params[2]; \
    uint k = params[3], metric = params[4]; \
    bool want_min = (metric == 0); /* L2: ascending; IP: descending */ \
    if (qi >= nq || k == 0) return; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    \
    /* Cooperative load of query into threadgroup memory */ \
    for (uint i = tid; i < d; i += TG_SIZE) \
        tgQ[i] = queries[qi * d + i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Strided scan: each thread computes distances on the fly, keeps local top-R */ \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    \
    for (uint j = tid; j < nb; j += TG_SIZE) { \
        const device float* vec = vectors + j * d; \
        float dist = 0.0f; \
        if (metric == 0) { \
            for (uint dd = 0; dd < d; dd++) { \
                float diff = tgQ[dd] - vec[dd]; \
                dist += diff * diff; \
            } \
        } else { \
            for (uint dd = 0; dd < d; dd++) \
                dist += tgQ[dd] * vec[dd]; \
        } \
        \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && dist < localDist[pos-1]) || (!want_min && dist > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; \
                localIdx[pos] = localIdx[pos-1]; \
                pos--; \
            } \
            localDist[pos] = dist; \
            localIdx[pos] = (int)j; \
            localCount++; \
        } else { \
            bool better = want_min ? (dist < localDist[R-1]) : (dist > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && dist < localDist[pos-1]) || (!want_min && dist > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; \
                    localIdx[pos] = localIdx[pos-1]; \
                    pos--; \
                } \
                localDist[pos] = dist; \
                localIdx[pos] = (int)j; \
            } \
        } \
    } \
    \
    /* Dump local candidates to threadgroup memory */ \
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
    /* Bitonic sort */ \
    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) { \
                uint partner = idx ^ j; \
                if (partner < CANDIDATES && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min \
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min \
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
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
    /* Write results */ \
    for (uint i = tid; i < K_out; i += TG_SIZE) { \
        outDistances[qi * k + i] = tgDist[i]; \
        outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < k - K_out; i += TG_SIZE) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + K_out + i] = -1; \
    } \
}
FUSED_DIST_TOPK_VARIANT(32)
FUSED_DIST_TOPK_VARIANT(64)
FUSED_DIST_TOPK_VARIANT(128)
FUSED_DIST_TOPK_VARIANT(256)
FUSED_DIST_TOPK_VARIANT(512)
FUSED_DIST_TOPK_VARIANT(1024)
#undef FUSED_DIST_TOPK_VARIANT

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
    constexpr uint R_out = R;  /* write all per-thread candidates to avoid birthday-problem misses */ \
    constexpr uint CANDIDATES = TG_SIZE * R_out;  /* 1024 */ \
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

// 2048 variant: R=16 per thread → 256×16 = 4096 candidate slots (32 KB threadgroup mem).
// Over-provisions to 4096 candidates so the strided scan never drops valid top-2048
// entries even when nb is close to k.  Direct-load path when nb ≤ 4096.
kernel void topk_threadgroup_2048(
    device const float* distances [[buffer(0)]],
    device float* outDistances [[buffer(1)]],
    device int* outIndices [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint qi [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint R = 16;
    constexpr uint CANDIDATES = TG_SIZE * R; // 4096
    threadgroup float tgDist[CANDIDATES];
    threadgroup int tgIdx[CANDIDATES];
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3];
    if (qi >= nq || k == 0) return;
    const device float* row = distances + qi * nb;
    uint kk = min(k, nb);
    uint K_out = min((uint)2048, kk);

    if (nb <= CANDIDATES) {
        for (uint i = tid; i < CANDIDATES; i += TG_SIZE) {
            if (i < nb) {
                tgDist[i] = row[i];
                tgIdx[i] = (int)i;
            } else {
                tgDist[i] = want_min ? 1e38f : -1e38f;
                tgIdx[i] = -1;
            }
        }
    } else {
        float localDist[R];
        int localIdx[R];
        uint localCount = 0;

        for (uint j = tid; j < nb; j += TG_SIZE) {
            float v = row[j];
            if (localCount < R) {
                uint pos = localCount;
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) {
                    localDist[pos] = localDist[pos-1];
                    localIdx[pos] = localIdx[pos-1];
                    pos--;
                }
                localDist[pos] = v;
                localIdx[pos] = (int)j;
                localCount++;
            } else {
                bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]);
                if (better) {
                    uint pos = R - 1;
                    while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) {
                        localDist[pos] = localDist[pos-1];
                        localIdx[pos] = localIdx[pos-1];
                        pos--;
                    }
                    localDist[pos] = v;
                    localIdx[pos] = (int)j;
                }
            }
        }

        for (uint i = 0; i < R; i++) {
            uint idx = tid * R + i;
            if (i < localCount) {
                tgDist[idx] = localDist[i];
                tgIdx[idx] = localIdx[i];
            } else {
                tgDist[idx] = want_min ? 1e38f : -1e38f;
                tgIdx[idx] = -1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CANDIDATES && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool partnerBetter = want_min ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                                      : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool idxBetter = want_min ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                                  : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    bool swap = ascending ? partnerBetter : idxBetter;
                    if (swap) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < K_out; i += TG_SIZE) {
        outDistances[qi * k + i] = tgDist[i];
        outIndices[qi * k + i] = tgIdx[i];
    }
    for (uint i = tid; i < k - K_out; i += TG_SIZE) {
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f;
        outIndices[qi * k + K_out + i] = -1;
    }
}

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

// ============================================================
// Pre-computed norms for L2 distance: ||a-b||² = ||a||² + ||b||² - 2·a·b
// compute_norms: computes ||v||² for each vector. Grid: (nb, 1).
// l2_with_norms: uses pre-computed vector norms + IP to compute L2.
// ============================================================

kernel void compute_norms(
    device const float* vectors [[buffer(0)]],
    device float*       norms   [[buffer(1)]],
    device const uint*  params  [[buffer(2)]],  // nb, d
    uint vid [[thread_position_in_grid]]
) {
    uint nb = params[0], d = params[1];
    if (vid >= nb) return;
    const device float* v = vectors + vid * d;
    float sum = 0.0f;
    uint d4 = d / 4;
    const device float4* v4 = (const device float4*)v;
    for (uint t = 0; t < d4; t++) {
        sum += dot(v4[t], v4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        sum += v[t] * v[t];
    }
    norms[vid] = sum;
}

// L2 distance using pre-computed vector (centroid) norms:
// dist[i][j] = queryNorm[i] + vecNorm[j] - 2 * dot(query[i], vec[j])
// We compute queryNorm on-the-fly (cheap, nq rows) but reuse vecNorms.
kernel void l2_with_norms(
    device const float* queries    [[buffer(0)]],
    device const float* vectors    [[buffer(1)]],
    device float*       distances  [[buffer(2)]],
    device const uint*  params     [[buffer(3)]],  // nq, nb, d
    device const float* vecNorms   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;

    const device float* q = queries + i * d;
    const device float* v = vectors + j * d;
    float dot_val = 0.0f;
    uint d4 = d / 4;
    const device float4* q4 = (const device float4*)q;
    const device float4* v4 = (const device float4*)v;
    for (uint t = 0; t < d4; t++) {
        dot_val += dot(q4[t], v4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        dot_val += q[t] * v[t];
    }

    // Compute query norm inline (avoids separate query norm buffer).
    float qNorm = 0.0f;
    for (uint t = 0; t < d4; t++) {
        qNorm += dot(q4[t], q4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        qNorm += q[t] * q[t];
    }

    distances[i * nb + j] = qNorm + vecNorms[j] - 2.0f * dot_val;
}

// ============================================================
// IVF Flat scan — two-pass design (mirrors CUDA IVF):
//
//   Pass 1  ivf_scan_list
//       Grid: (nq * nprobe) threadgroups, 256 threads each.
//       Each threadgroup scans ONE inverted list for one query
//       and writes a per-list top-k.
//       Output: perListDist/perListIdx — (nq * nprobe * k).
//
//   Pass 2  ivf_merge_lists
//       Grid: (nq) threadgroups, 256 threads each.
//       Merges nprobe per-list top-k into final top-k per query.
//
// Both use float4 vectorised loads for memory throughput.
// ============================================================
// Shared params layout (device const uint*):
//   [0] nq   [1] d   [2] k   [3] nprobe   [4] want_min

kernel void ivf_scan_list(
    device const float*      queries       [[buffer(0)]],
    device const float*      codes         [[buffer(1)]],
    device const long*  ids           [[buffer(2)]],
    device const uint*  listOffset    [[buffer(3)]],
    device const uint*  listLength    [[buffer(4)]],
    device const int*   coarseAssign  [[buffer(5)]],
    device       float* perListDist   [[buffer(6)]],
    device       long*  perListIdx    [[buffer(7)]],
    device const uint*  params        [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    // Cache query vector in threadgroup memory (read once from device, reused
    // by all threads for every vector in this list).
    threadgroup float tgQuery[512]; // max d supported; only first d floats used
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sort internally by (dist, vecIdx) using 32-bit vecIdx for speed;
    // resolve to 64-bit ID only when writing output.
    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint d4 = d / 4;

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device float* vvec = codes + vecIdx * d;

        float dist = 0.0f;
        if (want_min) {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                float4 diff = q4[t] - v4[t];
                dist += dot(diff, diff);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float diff = tgQuery[t] - vvec[t];
                dist += diff * diff;
            }
        } else {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                dist += dot(q4[t], v4[t]);
            }
            for (uint t = d4 * 4; t < d; t++) {
                dist += tgQuery[t] * vvec[t];
            }
        }

        int vi = (int)vecIdx;

        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K; // 1024
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output: resolve 32-bit vecIdx → 64-bit ID from ids buffer.
    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Small-list variant: 32 threads, 32-element bitonic sort ----
// Used when avg list size ≤ 32 (most threads in 256-thread version idle).
// Saves ~90% of bitonic sort barriers and threadgroup memory.
kernel void ivf_scan_list_small(
    device const float*      queries       [[buffer(0)]],
    device const float*      codes         [[buffer(1)]],
    device const long*  ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*  perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 32;
    constexpr uint LOCAL_K = 1;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    // Query cache in threadgroup memory.
    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestDist = sentinel;
    int   bestIdx  = -1;
    uint d4 = d / 4;

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device float* vvec = codes + vecIdx * d;

        float dist = 0.0f;
        if (want_min) {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                float4 diff = q4[t] - v4[t];
                dist += dot(diff, diff);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float diff = tgQuery[t] - vvec[t];
                dist += diff * diff;
            }
        } else {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                dist += dot(q4[t], v4[t]);
            }
            for (uint t = d4 * 4; t < d; t++) {
                dist += tgQuery[t] * vvec[t];
            }
        }

        int vi = (int)vecIdx;

        bool better = want_min
            ? (dist < bestDist || (dist == bestDist && vi < bestIdx))
            : (dist > bestDist || (dist == bestDist && vi < bestIdx));
        if (better) {
            bestDist = dist;
            bestIdx  = vi;
        }
    }

    constexpr uint CAND = TG_SIZE; // 32
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    tgDist[tid] = bestDist;
    tgIdx [tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            uint partner = tid ^ j;
            if (partner < CAND && partner > tid) {
                bool ascending = ((tid & k2) == 0);
                bool pB = want_min
                    ? (tgDist[partner] < tgDist[tid] || (tgDist[partner] == tgDist[tid] && tgIdx[partner] < tgIdx[tid]))
                    : (tgDist[partner] > tgDist[tid] || (tgDist[partner] == tgDist[tid] && tgIdx[partner] < tgIdx[tid]));
                bool iB = want_min
                    ? (tgDist[tid] < tgDist[partner] || (tgDist[tid] == tgDist[partner] && tgIdx[tid] < tgIdx[partner]))
                    : (tgDist[tid] > tgDist[partner] || (tgDist[tid] == tgDist[partner] && tgIdx[tid] < tgIdx[partner]));
                if (ascending ? pB : iB) {
                    float td = tgDist[tid]; tgDist[tid] = tgDist[partner]; tgDist[partner] = td;
                    int   ti = tgIdx [tid]; tgIdx [tid] = tgIdx [partner]; tgIdx [partner] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Interleaved layout scan: 32-vector blocks, dimensions interleaved ----
// Memory layout per block: [v0d0 v1d0 .. v31d0] [v0d1 v1d1 .. v31d1] ...
// Each simdgroup (32 threads) processes one block cooperatively,
// achieving coalesced reads: all threads read the same dimension simultaneously.
kernel void ivf_scan_list_interleaved(
    device const float*      queries           [[buffer(0)]],
    device const float*      codes             [[buffer(1)]],
    device const long*       ids               [[buffer(2)]],
    device const uint*       listOffset        [[buffer(3)]],
    device const uint*       listLength        [[buffer(4)]],
    device const int*        coarseAssign      [[buffer(5)]],
    device       float*      perListDist       [[buffer(6)]],
    device       long*       perListIdx        [[buffer(7)]],
    device const uint*       params            [[buffer(8)]],
    device const uint*       ilCodesOffset     [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE   = 256;
    constexpr uint SIMD_W    = 32;
    constexpr uint NUM_SIMDS = TG_SIZE / SIMD_W; // 8
    constexpr uint LOCAL_K   = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint idOff = listOffset[(uint)list_no];
    uint lLen  = listLength[(uint)list_no];
    uint cOff  = ilCodesOffset[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint numBlocks = (lLen + SIMD_W - 1) / SIMD_W;
    uint laneId    = tid % SIMD_W;
    uint simdId    = tid / SIMD_W;

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    for (uint blk = simdId; blk < numBlocks; blk += NUM_SIMDS) {
        uint vecInList = blk * SIMD_W + laneId;
        bool valid = vecInList < lLen;

        const device float* blockPtr = codes + cOff + blk * SIMD_W * d;

        float dist = 0.0f;
        if (want_min) {
            for (uint dd = 0; dd < d; dd += 4) {
                float v0 = valid ? blockPtr[(dd + 0) * SIMD_W + laneId] : 0.0f;
                float v1 = valid ? blockPtr[(dd + 1) * SIMD_W + laneId] : 0.0f;
                float v2 = valid ? blockPtr[(dd + 2) * SIMD_W + laneId] : 0.0f;
                float v3 = valid ? blockPtr[(dd + 3) * SIMD_W + laneId] : 0.0f;
                float4 diff = float4(
                    tgQuery[dd]     - v0,
                    tgQuery[dd + 1] - v1,
                    tgQuery[dd + 2] - v2,
                    tgQuery[dd + 3] - v3);
                dist += dot(diff, diff);
            }
        } else {
            for (uint dd = 0; dd < d; dd += 4) {
                float v0 = valid ? blockPtr[(dd + 0) * SIMD_W + laneId] : 0.0f;
                float v1 = valid ? blockPtr[(dd + 1) * SIMD_W + laneId] : 0.0f;
                float v2 = valid ? blockPtr[(dd + 2) * SIMD_W + laneId] : 0.0f;
                float v3 = valid ? blockPtr[(dd + 3) * SIMD_W + laneId] : 0.0f;
                dist += tgQuery[dd]     * v0
                      + tgQuery[dd + 1] * v1
                      + tgQuery[dd + 2] * v2
                      + tgQuery[dd + 3] * v3;
            }
        }

        if (!valid) dist = sentinel;
        int vi = valid ? (int)(idOff + vecInList) : -1;

        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K; // 1024
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Pass 2: merge nprobe per-list results per query ----
// Strided-scan pattern: each thread scans every TG_SIZE-th candidate across
// all nprobe×k entries, keeping its best LOCAL_K in registers. Then dump to
// threadgroup memory and bitonic-sort. Handles any nprobe×k without overflow.
kernel void ivf_merge_lists(
    device const float*      perListDist  [[buffer(0)]],
    device const long*       perListIdx   [[buffer(1)]],
    device       float*      outDistances [[buffer(2)]],
    device       long*       outIndices   [[buffer(3)]],
    device const uint*       params       [[buffer(4)]],
    uint qi  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE  = 256;
    constexpr uint LOCAL_K  = 4;
    constexpr uint CAND     = TG_SIZE * LOCAL_K; // 1024

    uint nq       = params[0];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint totalCand = nprobe * k;
    uint inputBase = qi * totalCand;

    float localDist[LOCAL_K];
    long  localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1L;
    }

    for (uint i = tid; i < totalCand; i += TG_SIZE) {
        float d = perListDist[inputBase + i];
        long  v = perListIdx [inputBase + i];

        bool better = want_min ? (d < localDist[LOCAL_K-1])
                               : (d > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            if (v < 0 && localCount >= LOCAL_K) continue;
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = d;
            localIdx [pos] = v;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                long  ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    threadgroup float tgDist[CAND];
    threadgroup long  tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1L;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        long  ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        outDistances[qi * k + i] = tgDist[i];
        outIndices  [qi * k + i] = tgIdx [i];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        outDistances[qi * k + kk + i] = sentinel;
        outIndices  [qi * k + kk + i] = -1L;
    }
}

// ============================================================
//  IVF Scalar Quantizer scan kernels
// ============================================================
//
// QT_8bit:  codes are uchar[d] per vector; decode: vmin[dim] + code * vdiff[dim]
//           SQ tables buffer layout: vmin[0..d-1], vdiff[0..d-1] (2*d floats)
// QT_4bit:  packed 2 values per byte, low/high nibble
// QT_6bit:  packed 4 values per 3 bytes
// QT_fp16:  codes are half[d] per vector; no SQ tables needed
//
// Both share the same top-k selection as ivf_scan_list.

inline float sq4_decode_component(device const uchar* code, uint i) {
    uchar b = code[i >> 1];
    uchar bits = (i & 1u) ? (b >> 4) : (b & 0x0Fu);
    return (float(bits) + 0.5f) / 15.0f;
}

inline float sq6_decode_component(device const uchar* code, uint i) {
    const device uchar* p = code + (i >> 2) * 3;
    uchar bits = 0;
    switch (i & 3u) {
        case 0:
            bits = p[0] & 0x3Fu;
            break;
        case 1:
            bits = (p[0] >> 6) | ((p[1] & 0x0Fu) << 2);
            break;
        case 2:
            bits = (p[1] >> 4) | ((p[2] & 0x03u) << 4);
            break;
        default:
            bits = p[2] >> 2;
            break;
    }
    return (float(bits) + 0.5f) / 63.0f;
}

kernel void ivf_scan_list_sq4(
    device const float*      queries       [[buffer(0)]],
    device const uchar*      codes         [[buffer(1)]],
    device const long*       ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*       perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    device const float*      sqTables      [[buffer(9)]],
    device const float*      centroids     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];
    uint by_residual = params[5];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }

    threadgroup float tgVmin[512];
    threadgroup float tgVdiff[512];
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgVmin [i] = sqTables[i];
        tgVdiff[i] = sqTables[d + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint codeSize = (d + 1) >> 1;
    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device uchar* cvec = codes + vecIdx * codeSize;

        float dist = 0.0f;
        for (uint t = 0; t < d; t++) {
            float xi = sq4_decode_component(cvec, t);
            float decoded = tgVmin[t] + xi * tgVdiff[t];
            if (by_residual) {
                decoded += centroids[(uint)list_no * d + t];
            }
            if (want_min) {
                float diff = tgQuery[t] - decoded;
                dist += diff * diff;
            } else {
                dist += tgQuery[t] * decoded;
            }
        }

        int vi = (int)vecIdx;
        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

kernel void ivf_scan_list_sq6(
    device const float*      queries       [[buffer(0)]],
    device const uchar*      codes         [[buffer(1)]],
    device const long*       ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*       perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    device const float*      sqTables      [[buffer(9)]],
    device const float*      centroids     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];
    uint by_residual = params[5];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }

    threadgroup float tgVmin[512];
    threadgroup float tgVdiff[512];
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgVmin [i] = sqTables[i];
        tgVdiff[i] = sqTables[d + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint codeSize = (d * 6 + 7) >> 3;
    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device uchar* cvec = codes + vecIdx * codeSize;

        float dist = 0.0f;
        for (uint t = 0; t < d; t++) {
            float xi = sq6_decode_component(cvec, t);
            float decoded = tgVmin[t] + xi * tgVdiff[t];
            if (by_residual) {
                decoded += centroids[(uint)list_no * d + t];
            }
            if (want_min) {
                float diff = tgQuery[t] - decoded;
                dist += diff * diff;
            } else {
                dist += tgQuery[t] * decoded;
            }
        }

        int vi = (int)vecIdx;
        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

kernel void ivf_scan_list_sq8_direct(
    device const float*      queries       [[buffer(0)]],
    device const uchar*      codes         [[buffer(1)]],
    device const long*       ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*       perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    device const float*      centroids     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];
    uint by_residual = params[5];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device uchar* cvec = codes + vecIdx * d;

        float dist = 0.0f;
        for (uint t = 0; t < d; t++) {
            float decoded = float(cvec[t]);
            if (by_residual) {
                decoded += centroids[(uint)list_no * d + t];
            }
            if (want_min) {
                float diff = tgQuery[t] - decoded;
                dist += diff * diff;
            } else {
                dist += tgQuery[t] * decoded;
            }
        }

        int vi = (int)vecIdx;
        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

kernel void ivf_scan_list_sq8(
    device const float*      queries       [[buffer(0)]],
    device const uchar*      codes         [[buffer(1)]],
    device const long*       ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*       perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    device const float*      sqTables      [[buffer(9)]],
    device const float*      centroids     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];
    uint by_residual = params[5];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }

    threadgroup float tgVmin[512];
    threadgroup float tgVdiff[512];
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgVmin [i] = sqTables[i];
        tgVdiff[i] = sqTables[d + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device uchar* cvec = codes + vecIdx * d;

        float dist = 0.0f;
        for (uint t = 0; t < d; t++) {
            float xi = (float(cvec[t]) + 0.5f) / 255.0f;
            float decoded = tgVmin[t] + xi * tgVdiff[t];
            if (by_residual) {
                decoded += centroids[(uint)list_no * d + t];
            }
            if (want_min) {
                float diff = tgQuery[t] - decoded;
                dist += diff * diff;
            } else {
                dist += tgQuery[t] * decoded;
            }
        }

        int vi = (int)vecIdx;
        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

kernel void ivf_scan_list_fp16(
    device const float*      queries       [[buffer(0)]],
    device const half*       codes         [[buffer(1)]],
    device const long*       ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*       perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    device const float*      centroids     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];
    uint by_residual = params[5];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    threadgroup float tgQuery[512];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint d4 = d / 4;

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device half* cvec = codes + vecIdx * d;

        float dist = 0.0f;
        if (want_min) {
            const device half4* v4 = (const device half4*)cvec;
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            for (uint t = 0; t < d4; t++) {
                float4 fv = float4(v4[t]);
                if (by_residual) {
                    float4 cv = float4(
                            centroids[(uint)list_no * d + t * 4 + 0],
                            centroids[(uint)list_no * d + t * 4 + 1],
                            centroids[(uint)list_no * d + t * 4 + 2],
                            centroids[(uint)list_no * d + t * 4 + 3]);
                    fv += cv;
                }
                float4 diff = q4[t] - fv;
                dist += dot(diff, diff);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float decoded = float(cvec[t]);
                if (by_residual) {
                    decoded += centroids[(uint)list_no * d + t];
                }
                float diff = tgQuery[t] - decoded;
                dist += diff * diff;
            }
        } else {
            const device half4* v4 = (const device half4*)cvec;
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            for (uint t = 0; t < d4; t++) {
                float4 fv = float4(v4[t]);
                if (by_residual) {
                    float4 cv = float4(
                            centroids[(uint)list_no * d + t * 4 + 0],
                            centroids[(uint)list_no * d + t * 4 + 1],
                            centroids[(uint)list_no * d + t * 4 + 2],
                            centroids[(uint)list_no * d + t * 4 + 3]);
                    fv += cv;
                }
                dist += dot(q4[t], fv);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float decoded = float(cvec[t]);
                if (by_residual) {
                    decoded += centroids[(uint)list_no * d + t];
                }
                dist += tgQuery[t] * decoded;
            }
        }

        int vi = (int)vecIdx;
        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
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

// ============================================================
//  IVF-PQ scan kernel (8-bit codes)
// ============================================================
//
// One threadgroup per (query, probe). Layout matches ivf_scan_list:
// output is per-list top-k in perListDist/perListIdx.
//
// lookupTable layout: (nq * nprobe) * M * 256 floats
//   For entry (qi, probe), offset = (qi * nprobe + probe) * M * 256
//   table[m * 256 + code] gives the partial distance contribution.
//
// params: [nq, M, k, nprobe, want_min]

kernel void ivf_scan_list_pq8(
    device const float* lookupTable [[buffer(0)]],
    device const uchar* codes       [[buffer(1)]],
    device const long*  ids         [[buffer(2)]],
    device const uint*  listOffset  [[buffer(3)]],
    device const uint*  listLength  [[buffer(4)]],
    device const int*   coarseAssign [[buffer(5)]],
    device float*       perListDist  [[buffer(6)]],
    device long*        perListIdx   [[buffer(7)]],
    device const uint*  params       [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint M        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint tableOff = (qi * nprobe + pi) * M * 256;

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        device const uchar* cvec = codes + vecIdx * M;
        float dist = 0.0f;
        for (uint m = 0; m < M; m++) {
            dist += lookupTable[tableOff + m * 256 + uint(cvec[m])];
        }

        int vi = (int)vecIdx;

        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = want_min
                    ? (localDist[pos] < localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]))
                    : (localDist[pos] > localDist[pos-1] || (localDist[pos] == localDist[pos-1] && localIdx[pos] < localIdx[pos-1]));
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = want_min
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool iB = want_min
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ============================================================
//  Binary (Hamming) distance — brute-force top-k
// ============================================================
//
// One threadgroup per query. Threads stride over database vectors,
// computing Hamming distance via XOR + popcount on uint32 words.
//
// params: [nq, nb, code_size, k]

kernel void hamming_distance_topk(
    device const uchar* queries  [[buffer(0)]],
    device const uchar* database [[buffer(1)]],
    device int*         outDist  [[buffer(2)]],
    device long*        outIdx   [[buffer(3)]],
    device const uint*  params   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq        = params[0];
    uint nb        = params[1];
    uint code_size = params[2];
    uint k         = params[3];

    uint qi = tgid;
    if (qi >= nq || k == 0) return;

    constexpr int sentinel = 0x7FFFFFFF;

    threadgroup uchar tgQuery[512];
    device const uchar* qptr = queries + qi * code_size;
    for (uint i = tid; i < code_size; i += TG_SIZE)
        tgQuery[i] = qptr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int localDist[LOCAL_K];
    int localIdx [LOCAL_K];
    uint localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint code_size4 = code_size / 4;
    uint code_tail  = code_size4 * 4;

    for (uint vi = tid; vi < nb; vi += TG_SIZE) {
        device const uchar* vptr = database + vi * code_size;
        int dist = 0;

        device const uint* v32 = (device const uint*)vptr;
        const threadgroup uint* q32 = (const threadgroup uint*)tgQuery;
        for (uint w = 0; w < code_size4; w++)
            dist += popcount(q32[w] ^ v32[w]);

        for (uint b = code_tail; b < code_size; b++)
            dist += popcount(uint(tgQuery[b]) ^ uint(vptr[b]));

        bool better = (dist < localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = (int)vi;
            while (pos > 0) {
                bool sw = (localDist[pos] < localDist[pos-1]) ||
                          (localDist[pos] == localDist[pos-1] &&
                           localIdx[pos] < localIdx[pos-1]);
                if (!sw) break;
                int td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K;
    threadgroup int tgDistH[CAND];
    threadgroup int tgIdxH [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDistH[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdxH [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = (tgDistH[partner] < tgDistH[idx]) ||
                              (tgDistH[partner] == tgDistH[idx] && tgIdxH[partner] < tgIdxH[idx]);
                    bool iB = (tgDistH[idx] < tgDistH[partner]) ||
                              (tgDistH[idx] == tgDistH[partner] && tgIdxH[idx] < tgIdxH[partner]);
                    if (ascending ? pB : iB) {
                        int td = tgDistH[idx]; tgDistH[idx] = tgDistH[partner]; tgDistH[partner] = td;
                        int ti = tgIdxH [idx]; tgIdxH [idx] = tgIdxH [partner]; tgIdxH [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        outDist[qi * k + i] = tgDistH[i];
        outIdx [qi * k + i] = (long)tgIdxH[i];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        outDist[qi * k + kk + i] = sentinel;
        outIdx [qi * k + kk + i] = -1L;
    }
}
