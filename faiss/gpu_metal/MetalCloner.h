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
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndicesOptions.h>

namespace faiss {
namespace gpu_metal {

class StandardMetalResources;

/// Options controlling how CPU indexes are cloned to Metal GPU.
/// Mirrors faiss::gpu::GpuClonerOptions for the Metal backend.
struct MetalClonerOptions {
    /// How IVF labels are represented (CPU/IVF/32-bit/64-bit).
    faiss::gpu::IndicesOptions indicesOptions = faiss::gpu::INDICES_64_BIT;

    /// Store flat vectors as float16 (MetalIndexFlat only).
    bool useFloat16 = false;

    /// Store IVF coarse-quantizer centroids as float16.
    bool useFloat16CoarseQuantizer = false;

    /// Match CUDA cloner option for IVFPQ precomputed tables.
    bool usePrecomputed = false;

    /// Reserve space for this many vectors in IVF inverted lists.
    long reserveVecs = 0;

    /// Enable/disable interleaved IVF layout.
    bool interleavedLayout = true;

    /// For CPU IndexIVFFlat input, convert list storage to IVF scalar
    /// quantization on Metal (instead of keeping full-float IVFFlat).
    bool useIVFScalarQuantizer = false;

    /// Quantizer type used when useIVFScalarQuantizer=true.
    faiss::ScalarQuantizer::QuantizerType ivfSQType =
            faiss::ScalarQuantizer::QT_8bit;

    /// Match CUDA option shape: accepted for API parity.
    /// Metal Flat does not use a transposed storage layout today.
    bool storeTransposed = false;

    /// If false, reject IVF indexes whose coarse quantizer is not IndexFlat.
    /// If true, allow cloner to reconstruct coarse centroids from the CPU
    /// quantizer and proceed.
    bool allowCpuCoarseQuantizer = false;

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
