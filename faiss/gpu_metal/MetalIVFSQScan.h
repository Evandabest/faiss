// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#import <Metal/Metal.h>
#include <cstdint>

namespace faiss::gpu_metal {
/// Synchronous direct scan entry point. Buffers are shared, sized by params:
/// nq,d,k,nprobe,nlist,fp16,residual,L2,code_size. IDs must be nonnegative;
/// offsets/lengths describe disjoint ranges inside codes/ids. Throws on
/// failure.
void runMetalIVFSQScan(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> queries,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> offsets,
        id<MTLBuffer> lengths,
        id<MTLBuffer> assignments,
        id<MTLBuffer> centroids,
        id<MTLBuffer> ranges,
        id<MTLBuffer> distances,
        id<MTLBuffer> labels,
        id<MTLBuffer> coarseScores,
        const uint32_t* params);
} // namespace faiss::gpu_metal
