// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MSL kernels: L2 squared / IP distance matrix, then top-k reduction.
 */

#import "MetalFlatKernels.h"
#include <cstring>

namespace faiss {
namespace gpu_metal {

namespace {

static const char* kMSLSource = R"msl(
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

// Fixed-size top-k variants: one kernel per K in {32,64,128,256,512,1024,2048}.
// Threadgroup arrays are local to the kernel so the compiler knows the size.
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
)msl";

// Maximum k supported (fits in 16 KB threadgroup memory: 8*k bytes).
static constexpr int kMaxK = 2048;

// Variant sizes: pick smallest K >= k. Order must match kTopKVariantNames.
static const int kTopKVariantSizes[] = {32, 64, 128, 256, 512, 1024, 2048};
static const int kNumTopKVariants = sizeof(kTopKVariantSizes) / sizeof(kTopKVariantSizes[0]);
static const char* kTopKVariantNames[] = {
    "topk_heap_32", "topk_heap_64", "topk_heap_128", "topk_heap_256",
    "topk_heap_512", "topk_heap_1024", "topk_heap_2048"
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

bool runFlatSearchGPU(
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

    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kMSLSource) options:nil error:&err];
    if (!lib) {
        return false;
    }

    id<MTLFunction> fnDist = [lib newFunctionWithName:isL2 ? @"l2_squared_matrix" : @"ip_matrix"];
    int variantIndex = selectTopKVariant(k);
    id<MTLFunction> fnTopK = [lib newFunctionWithName:@(kTopKVariantNames[variantIndex])];
    if (!fnDist || !fnTopK) {
        return false;
    }

    id<MTLComputePipelineState> psDist = [device newComputePipelineStateWithFunction:fnDist error:&err];
    id<MTLComputePipelineState> psTopK = [device newComputePipelineStateWithFunction:fnTopK error:&err];
    if (!psDist || !psTopK) {
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // Distance matrix
    const NSUInteger w = 16;
    const NSUInteger h = 16;
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

    // Top-k: fixed-size variant (no setThreadgroupMemoryLength; kernel declares size)
    [enc setComputePipelineState:psTopK];
    [enc setBuffer:distMatrix offset:0 atIndex:0];
    [enc setBuffer:outDistances offset:0 atIndex:1];
    [enc setBuffer:outIndices offset:0 atIndex:2];
    uint32_t topkArgs[4] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)k, isL2 ? 1u : 0u};
    [enc setBytes:topkArgs length:sizeof(topkArgs) atIndex:3];
    MTLSize gridTopK = MTLSizeMake(nq, 1, 1);
    [enc dispatchThreadgroups:gridTopK threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return true;
}

int getMetalFlatSearchMaxK() {
    return kMaxK;
}

} // namespace gpu_metal
} // namespace faiss
