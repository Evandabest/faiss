// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalCloner.h"
#import "StandardMetalResources.h"
#import "MetalIndexFlat.h"
#import "MetalIndexIVFFlat.h"
#import "MetalIndexIVFScalarQuantizer.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>

namespace faiss {
namespace gpu_metal {

int get_num_gpus() {
    auto res = std::make_shared<MetalResources>();
    return res->isAvailable() ? 1 : 0;
}

faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index) {
    FAISS_THROW_IF_NOT(res != nullptr);
    FAISS_THROW_IF_NOT(res->getResources() != nullptr);
    FAISS_THROW_IF_NOT(res->getResources()->isAvailable());
    FAISS_THROW_IF_NOT_MSG(device == 0, "Metal backend supports only device 0");

    MetalIndexConfig config;
    config.device = 0;

    // IndexIVFScalarQuantizer (check before IndexIVFFlat)
    const auto* ivfSQ = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index);
    if (ivfSQ) {
        FAISS_THROW_IF_NOT(
                ivfSQ->metric_type == METRIC_L2 ||
                ivfSQ->metric_type == METRIC_INNER_PRODUCT);
        auto* metal = new MetalIndexIVFScalarQuantizer(
                res->getResources(), ivfSQ, config);
        return metal;
    }

    // IndexIVFFlat (check before IndexFlat since IVFFlat's quantizer is IndexFlat)
    const auto* ivfFlat = dynamic_cast<const faiss::IndexIVFFlat*>(index);
    if (ivfFlat) {
        FAISS_THROW_IF_NOT(
                ivfFlat->metric_type == METRIC_L2 ||
                ivfFlat->metric_type == METRIC_INNER_PRODUCT);
        auto* metal = new MetalIndexIVFFlat(
                res->getResources(), ivfFlat, config);
        return metal;
    }

    const auto* flat = dynamic_cast<const faiss::IndexFlat*>(index);
    if (flat) {
        FAISS_THROW_IF_NOT(
                flat->metric_type == METRIC_L2 ||
                flat->metric_type == METRIC_INNER_PRODUCT);
        auto* metal = new MetalIndexFlat(
                res->getResources(),
                flat->d,
                flat->metric_type,
                flat->metric_arg,
                config);
        metal->copyFrom(flat);
        return metal;
    }

    FAISS_THROW_MSG(
            "index_cpu_to_metal_gpu: unsupported index type "
            "(only IndexFlat, IndexIVFFlat, and IndexIVFScalarQuantizer supported)");
}

faiss::Index* index_metal_gpu_to_cpu(const faiss::Index* index) {
    const auto* metalIVFSQ =
            dynamic_cast<const MetalIndexIVFScalarQuantizer*>(index);
    if (metalIVFSQ) {
        faiss::IndexFlat* quantizer =
                (metalIVFSQ->metric_type == METRIC_INNER_PRODUCT)
                ? (faiss::IndexFlat*)new faiss::IndexFlatIP(metalIVFSQ->d)
                : (faiss::IndexFlat*)new faiss::IndexFlatL2(metalIVFSQ->d);
        auto* cpu = new faiss::IndexIVFScalarQuantizer(
                quantizer,
                metalIVFSQ->d,
                metalIVFSQ->nlist(),
                metalIVFSQ->sqQuantizerType(),
                metalIVFSQ->metric_type);
        cpu->own_fields = true;
        metalIVFSQ->copyTo(cpu);
        return cpu;
    }

    const auto* metalIVF = dynamic_cast<const MetalIndexIVFFlat*>(index);
    if (metalIVF) {
        faiss::IndexFlat* quantizer =
                (metalIVF->metric_type == METRIC_INNER_PRODUCT)
                ? (faiss::IndexFlat*)new faiss::IndexFlatIP(metalIVF->d)
                : (faiss::IndexFlat*)new faiss::IndexFlatL2(metalIVF->d);
        auto* cpu = new faiss::IndexIVFFlat(
                quantizer,
                metalIVF->d,
                metalIVF->nlist(),
                metalIVF->metric_type);
        cpu->own_fields = true;
        metalIVF->copyTo(cpu);
        return cpu;
    }

    const auto* metalFlat = dynamic_cast<const MetalIndexFlat*>(index);
    if (metalFlat) {
        faiss::IndexFlat* cpu =
                (metalFlat->metric_type == METRIC_INNER_PRODUCT)
                ? (faiss::IndexFlat*)new faiss::IndexFlatIP(metalFlat->d)
                : (faiss::IndexFlat*)new faiss::IndexFlatL2(metalFlat->d);
        cpu->metric_arg = metalFlat->metric_arg;
        metalFlat->copyTo(cpu);
        return cpu;
    }

    FAISS_THROW_MSG(
            "index_metal_gpu_to_cpu: unsupported index type "
            "(only MetalIndexFlat, MetalIndexIVFFlat, and "
            "MetalIndexIVFScalarQuantizer supported)");
}

} // namespace gpu_metal
} // namespace faiss
