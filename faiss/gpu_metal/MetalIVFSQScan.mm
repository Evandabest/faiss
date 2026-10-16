// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/gpu_metal/MetalIVFSQScan.h>
#include <faiss/gpu_metal/MetalKernels.h>
#include <faiss/impl/FaissAssert.h>
#include <limits>

namespace faiss::gpu_metal {
namespace {
void requireBytes(id<MTLBuffer> buffer, size_t a, size_t b, size_t width) {
    FAISS_THROW_IF_NOT(a == 0 || b <= SIZE_MAX / a);
    size_t count = a * b;
    FAISS_THROW_IF_NOT(count <= SIZE_MAX / width);
    FAISS_THROW_IF_NOT(buffer && buffer.length >= count * width);
}
struct EncoderGuard {
    id<MTLComputeCommandEncoder> encoder;
    ~EncoderGuard() { if (encoder) [encoder endEncoding]; }
};
}

void runMetalIVFSQScan(
        id<MTLDevice> device, id<MTLCommandQueue> queue,
        id<MTLBuffer> queries, id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> offsets, id<MTLBuffer> lengths,
        id<MTLBuffer> assignments, id<MTLBuffer> centroids,
        id<MTLBuffer> ranges, id<MTLBuffer> distances, id<MTLBuffer> labels,
        const uint32_t* p) {
    FAISS_THROW_IF_NOT(device && queue && p);
    FAISS_THROW_IF_NOT(p[0] && p[1] && p[2] && p[2] <= 512 && p[3] && p[4]);
    FAISS_THROW_IF_NOT(uint64_t(p[1]) * (p[5] ? 2 : 1) == p[8]);
    requireBytes(queries, p[0], p[1], 4);
    requireBytes(assignments, p[0], p[3], 4);
    requireBytes(offsets, p[4], 1, 4);
    requireBytes(lengths, p[4], 1, 4);
    requireBytes(centroids, p[4], p[6] ? p[1] : 0, 4);
    requireBytes(ranges, p[5] ? 0 : 2, p[1], 4);
    requireBytes(distances, p[0], p[2], 4);
    requireBytes(labels, p[0], p[2], 8);
    const auto* off = (const uint32_t*)offsets.contents;
    const auto* len = (const uint32_t*)lengths.contents;
    for (size_t l = 0; l < p[4]; ++l) {
        size_t end = size_t(off[l]) + len[l];
        requireBytes(codes, end, p[8], 1);
        requireBytes(ids, end, 1, 8);
    }
    auto& kernels = getMetalKernels(device);
    FAISS_THROW_IF_NOT(kernels.isValid());
    id<MTLCommandBuffer> command = [queue commandBuffer];
    FAISS_THROW_IF_NOT(command);
    EncoderGuard guard{[command computeCommandEncoder]};
    FAISS_THROW_IF_NOT(guard.encoder);
    id<MTLBuffer> __unsafe_unretained buffers[] = {queries, codes, ids, offsets, lengths,
        assignments, centroids, ranges, distances, labels};
    kernels.encodeIVFSQScan(guard.encoder, buffers, p);
    [guard.encoder endEncoding];
    guard.encoder = nil;
    [command commit];
    [command waitUntilCompleted];
    FAISS_THROW_IF_NOT_MSG(command.status == MTLCommandBufferStatusCompleted,
                          "Metal IVFSQ command execution failed");
}
} // namespace faiss::gpu_metal
