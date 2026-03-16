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
#import "MetalIndexIVFPQ.h"
#import "MetalIndexBinaryFlat.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<faiss::IndexIVFScalarQuantizer> convertIVFFlatToIVFSQ(
        const faiss::IndexIVFFlat* src,
        faiss::ScalarQuantizer::QuantizerType sqType) {
    FAISS_THROW_IF_NOT_MSG(src, "convertIVFFlatToIVFSQ: source index is null");
    FAISS_THROW_IF_NOT_MSG(src->quantizer, "convertIVFFlatToIVFSQ: source quantizer is null");
    FAISS_THROW_IF_NOT_MSG(src->is_trained, "convertIVFFlatToIVFSQ: source index is not trained");

    faiss::IndexFlat* dstQuantizer =
            (src->metric_type == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(src->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(src->d);
    auto dst = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            dstQuantizer,
            src->d,
            src->nlist,
            sqType,
            src->metric_type,
            /*by_residual=*/false);
    dst->own_fields = true;
    dst->metric_arg = src->metric_arg;

    // Copy coarse centroids so list assignment behavior matches the source IVF.
    if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * src->d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        if (!dst->quantizer->is_trained) {
            dst->quantizer->train(src->nlist, coarse.data());
        }
        dst->quantizer->add(src->nlist, coarse.data());
    }

    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l) {
        totalN += src->invlists->list_size(l);
    }

    std::vector<float> allVecs;
    std::vector<faiss::idx_t> allIds;
    if (totalN > 0) {
        allVecs.resize(totalN * (size_t)src->d);
        allIds.resize(totalN);

        size_t pos = 0;
        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0) {
                continue;
            }
            const uint8_t* codes = src->invlists->get_codes(l);
            const faiss::idx_t* ids = src->invlists->get_ids(l);
            std::memcpy(
                    allVecs.data() + pos * (size_t)src->d,
                    codes,
                    ls * (size_t)src->d * sizeof(float));
            std::memcpy(allIds.data() + pos, ids, ls * sizeof(faiss::idx_t));
            pos += ls;
        }
    }

    // Train SQ codec from vectors when available; for empty indexes train from
    // coarse centroids so the destination remains usable.
    if (totalN > 0) {
        dst->train((faiss::idx_t)totalN, allVecs.data());
    } else if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * src->d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        dst->train(src->nlist, coarse.data());
    }

    if (totalN > 0) {
        dst->add_with_ids((faiss::idx_t)totalN, allVecs.data(), allIds.data());
    }

    dst->nprobe = src->nprobe;
    return dst;
}

} // namespace

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
    return index_cpu_to_metal_gpu(res, device, index, nullptr);
}

faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options) {
    FAISS_THROW_IF_NOT(res != nullptr);
    FAISS_THROW_IF_NOT(res->getResources() != nullptr);
    FAISS_THROW_IF_NOT(res->getResources()->isAvailable());
    FAISS_THROW_IF_NOT_MSG(device == 0, "Metal backend supports only device 0");

    MetalClonerOptions opts;
    if (options) {
        opts = *options;
    }

    MetalIndexConfig config;
    config.device = 0;
    config.useFloat16 = opts.useFloat16;
    config.useFloat16CoarseQuantizer = opts.useFloat16CoarseQuantizer;
    config.indicesOptions = opts.indicesOptions;
    config.interleavedLayout = opts.interleavedLayout;
    config.storeTransposed = opts.storeTransposed;

    auto coarseQuantizerAllowed = [&](const faiss::IndexIVF* ivf) {
        if (!ivf || !ivf->quantizer) {
            return;
        }
        const bool isFlat =
                dynamic_cast<const faiss::IndexFlat*>(ivf->quantizer) != nullptr;
        FAISS_THROW_IF_NOT_MSG(
                isFlat || opts.allowCpuCoarseQuantizer,
                "index_cpu_to_metal_gpu: coarse quantizer must be IndexFlat unless "
                "allowCpuCoarseQuantizer=true");
    };

    // IndexIVFPQ (check before IndexIVFFlat)
    const auto* ivfPQ = dynamic_cast<const faiss::IndexIVFPQ*>(index);
    if (ivfPQ) {
        FAISS_THROW_IF_NOT(
                ivfPQ->metric_type == METRIC_L2 ||
                ivfPQ->metric_type == METRIC_INNER_PRODUCT);
        coarseQuantizerAllowed(ivfPQ);
        FAISS_THROW_IF_NOT_MSG(
                ivfPQ->pq.nbits == 8,
                "Metal IVFPQ only supports 8-bit PQ codes");
        auto* metal = new MetalIndexIVFPQ(
                res->getResources(), ivfPQ, config);
        metal->setUsePrecomputedTables(opts.usePrecomputed);
        metal->verbose = opts.verbose;
        if (opts.reserveVecs > 0) {
            metal->reserveMemory(opts.reserveVecs);
        }
        return metal;
    }

    // IndexIVFScalarQuantizer (check before IndexIVFFlat)
    const auto* ivfSQ = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index);
    if (ivfSQ) {
        FAISS_THROW_IF_NOT(
                ivfSQ->metric_type == METRIC_L2 ||
                ivfSQ->metric_type == METRIC_INNER_PRODUCT);
        coarseQuantizerAllowed(ivfSQ);
        auto* metal = new MetalIndexIVFScalarQuantizer(
                res->getResources(), ivfSQ, config);
        metal->verbose = opts.verbose;
        if (opts.reserveVecs > 0) {
            metal->reserveMemory(opts.reserveVecs);
        }
        return metal;
    }

    // IndexIVFFlat (check before IndexFlat since IVFFlat's quantizer is IndexFlat)
    const auto* ivfFlat = dynamic_cast<const faiss::IndexIVFFlat*>(index);
    if (ivfFlat) {
        FAISS_THROW_IF_NOT(
                ivfFlat->metric_type == METRIC_L2 ||
                ivfFlat->metric_type == METRIC_INNER_PRODUCT);
        coarseQuantizerAllowed(ivfFlat);

        if (opts.useIVFScalarQuantizer) {
            auto cpuSQ = convertIVFFlatToIVFSQ(ivfFlat, opts.ivfSQType);
            auto* metal = new MetalIndexIVFScalarQuantizer(
                    res->getResources(), cpuSQ.get(), config);
            metal->verbose = opts.verbose;
            if (opts.reserveVecs > 0 && ivfFlat->ntotal == 0) {
                metal->reserveMemory(opts.reserveVecs);
            }
            return metal;
        }

        auto* metal = new MetalIndexIVFFlat(
                res->getResources(), ivfFlat, config);
        metal->verbose = opts.verbose;
        if (opts.reserveVecs > 0) {
            metal->reserveMemory(opts.reserveVecs);
        }
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
        metal->verbose = opts.verbose;
        return metal;
    }

    FAISS_THROW_MSG(
            "index_cpu_to_metal_gpu: unsupported index type "
            "(supported: IndexFlat, IndexIVFFlat, IndexIVFScalarQuantizer, IndexIVFPQ)");
}

faiss::Index* index_metal_gpu_to_cpu(const faiss::Index* index) {
    const auto* metalIVFPQ = dynamic_cast<const MetalIndexIVFPQ*>(index);
    if (metalIVFPQ) {
        int M = metalIVFPQ->getNumSubQuantizers();
        faiss::IndexFlat* quantizer =
                (metalIVFPQ->metric_type == METRIC_INNER_PRODUCT)
                ? (faiss::IndexFlat*)new faiss::IndexFlatIP(metalIVFPQ->d)
                : (faiss::IndexFlat*)new faiss::IndexFlatL2(metalIVFPQ->d);
        auto* cpu = new faiss::IndexIVFPQ(
                quantizer, metalIVFPQ->d, metalIVFPQ->nlist(),
                (size_t)M, 8);
        cpu->own_fields = true;
        metalIVFPQ->copyTo(cpu);
        return cpu;
    }

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
            "(supported: MetalIndexFlat, MetalIndexIVFFlat, "
            "MetalIndexIVFScalarQuantizer, MetalIndexIVFPQ)");
}

faiss::IndexBinary* index_binary_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::IndexBinary* index) {
    FAISS_THROW_IF_NOT(res != nullptr);
    FAISS_THROW_IF_NOT(res->getResources() != nullptr);
    FAISS_THROW_IF_NOT(res->getResources()->isAvailable());
    FAISS_THROW_IF_NOT_MSG(device == 0, "Metal backend supports only device 0");

    const auto* binaryFlat =
            dynamic_cast<const faiss::IndexBinaryFlat*>(index);
    if (binaryFlat) {
        auto* metal = new MetalIndexBinaryFlat(
                res->getResources(), binaryFlat);
        return metal;
    }

    FAISS_THROW_MSG(
            "index_binary_cpu_to_metal_gpu: unsupported index type "
            "(supported: IndexBinaryFlat)");
}

faiss::IndexBinary* index_binary_metal_gpu_to_cpu(
        const faiss::IndexBinary* index) {
    const auto* metalBF =
            dynamic_cast<const MetalIndexBinaryFlat*>(index);
    if (metalBF) {
        auto* cpu = new faiss::IndexBinaryFlat(metalBF->d);
        metalBF->copyTo(cpu);
        return cpu;
    }

    FAISS_THROW_MSG(
            "index_binary_metal_gpu_to_cpu: unsupported index type "
            "(supported: MetalIndexBinaryFlat)");
}

} // namespace gpu_metal
} // namespace faiss
