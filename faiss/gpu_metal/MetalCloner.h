// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Clone CPU <-> Metal GPU. Mirrors GpuCloner roles for Metal backend.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>

namespace faiss {
namespace gpu_metal {

class StandardMetalResources;

/// Options controlling how CPU indexes are cloned to Metal GPU.
/// Mirrors faiss::gpu::GpuClonerOptions for the Metal backend.
struct MetalClonerOptions {
    /// Store flat vectors as float16 (MetalIndexFlat only).
    bool useFloat16 = false;

    /// Store IVF coarse-quantizer centroids as float16.
    bool useFloat16CoarseQuantizer = false;

    /// Reserve space for this many vectors in IVF inverted lists.
    long reserveVecs = 0;

    /// Set verbose flag on the created index.
    bool verbose = false;
};

/// Returns the number of Metal "devices" (1 if Metal is available, else 0).
int get_num_gpus();

/// Clone a CPU index to Metal GPU with default options.
faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index);

/// Clone a CPU index to Metal GPU with explicit options.
faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options);

/// Copy a Metal index back to CPU. Caller owns the returned index.
faiss::Index* index_metal_gpu_to_cpu(const faiss::Index* index);

/// Clone a CPU binary index to Metal GPU.
faiss::IndexBinary* index_binary_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::IndexBinary* index);

/// Copy a Metal binary index back to CPU. Caller owns the returned index.
faiss::IndexBinary* index_binary_metal_gpu_to_cpu(
        const faiss::IndexBinary* index);

} // namespace gpu_metal
} // namespace faiss
