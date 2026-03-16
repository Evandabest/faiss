// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFPQ.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/impl/MetalIVFPQ.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu_metal {

// ============================================================
//  Constructors
// ============================================================

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        int dims,
        idx_t nlist,
        int M,
        int nbitsPerIdx,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config) {
    FAISS_THROW_IF_NOT(nbitsPerIdx == 8);
    FAISS_THROW_IF_NOT(dims % M == 0);

    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
    cpuIndex_ = std::make_unique<faiss::IndexIVFPQ>(
            quantizer, (size_t)dims, (size_t)nlist, (size_t)M,
            (size_t)nbitsPerIdx);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFPQImpl>(
            resources, dims, nlist, M, nbitsPerIdx, metric, metricArg);
}

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexIVFPQ* cpuIndex,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  (int)cpuIndex->d,
                  cpuIndex->metric_type,
                  cpuIndex->metric_arg,
                  config) {
    FAISS_THROW_IF_NOT(cpuIndex->pq.nbits == 8);

    int M = (int)cpuIndex->pq.M;
    faiss::IndexFlat* quantizer = (cpuIndex->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)cpuIndex->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)cpuIndex->d);
    cpuIndex_ = std::make_unique<faiss::IndexIVFPQ>(
            quantizer, cpuIndex->d, cpuIndex->nlist, (size_t)M,
            cpuIndex->pq.nbits);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFPQImpl>(
            resources, (int)cpuIndex->d, cpuIndex->nlist, M,
            (int)cpuIndex->pq.nbits,
            cpuIndex->metric_type, cpuIndex->metric_arg);
    copyFrom(cpuIndex);
}

MetalIndexIVFPQ::~MetalIndexIVFPQ() = default;

// ============================================================
//  Helpers
// ============================================================

void MetalIndexIVFPQ::ensureSearchBuf_(
        id<MTLBuffer>& buf,
        size_t& cap,
        size_t needed) const {
    if (buf != nil && cap >= needed) return;
    size_t newCap = std::max(needed, cap * 2);
    id<MTLDevice> device = resources_->getDevice();
    buf = [device newBufferWithLength:newCap
                              options:MTLResourceStorageModeShared];
    cap = buf ? newCap : 0;
}

void MetalIndexIVFPQ::uploadCentroids_() const {
    if (!cpuIndex_ || !cpuIndex_->quantizer || !resources_) return;
    auto* flatQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    if (!flatQ || flatQ->ntotal == 0) {
        centroidBuf_ = nil;
        centroidNormsBuf_ = nil;
        return;
    }
    size_t nCentroids = (size_t)flatQ->ntotal;
    size_t bytes = nCentroids * (size_t)d * sizeof(float);
    id<MTLDevice> device = resources_->getDevice();
    if (!device) return;

    centroidBuf_ = [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    if (centroidBuf_) {
        std::memcpy([centroidBuf_ contents], flatQ->get_xb(), bytes);
    }
    centroidNormsBuf_ = nil;
}

void MetalIndexIVFPQ::uploadPQCentroids_() const {
    if (!cpuIndex_ || !gpuIvf_) return;
    const auto& pq = cpuIndex_->pq;
    if (pq.centroids.empty()) return;
    gpuIvf_->setPQCentroids(pq.centroids.data());
}

void MetalIndexIVFPQ::computeLookupTables_(
        int nq,
        const float* queries,
        int nprobe,
        const idx_t* coarseAssign,
        const float* coarseDist,
        float* tables) const {
    const auto& pq = cpuIndex_->pq;
    int M = (int)pq.M;
    int ksub = (int)pq.ksub;
    const bool isL2 = (metric_type == METRIC_L2);
    const int tableStride = M * ksub;
    const bool canUsePrecomputedL2 =
            isL2 && cpuIndex_->by_residual &&
            cpuIndex_->use_precomputed_table == 1 &&
            cpuIndex_->precomputed_table.size() >=
                    (size_t)cpuIndex_->nlist * (size_t)tableStride;

    auto* flatQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    const float* coarseCentroids = flatQ ? flatQ->get_xb() : nullptr;

#pragma omp parallel for if (nq * nprobe > 512)
    for (int qi = 0; qi < nq; ++qi) {
        const float* query = queries + qi * d;
        std::vector<float> residual((size_t)d);
        float* firstTab = tables + (size_t)qi * (size_t)nprobe * (size_t)tableStride;

        if (!isL2) {
            // IP tables do not depend on probe/list assignment.
            pq.compute_inner_prod_table(query, firstTab);
            for (int pi = 1; pi < nprobe; ++pi) {
                float* tab = firstTab + (size_t)pi * (size_t)tableStride;
                std::memcpy(tab, firstTab, (size_t)tableStride * sizeof(float));
            }
            continue;
        }

        if (canUsePrecomputedL2) {
            std::vector<float> sim2((size_t)tableStride);
            pq.compute_inner_prod_table(query, sim2.data());
            for (int pi = 0; pi < nprobe; ++pi) {
                idx_t listId = coarseAssign[qi * nprobe + pi];
                float* tab = firstTab + (size_t)pi * (size_t)tableStride;
                if (listId < 0) {
                    std::fill_n(tab, tableStride, 1e38f);
                    continue;
                }

                const float* pc = cpuIndex_->precomputed_table.data() +
                        (size_t)listId * (size_t)tableStride;
                const float coarse = coarseDist ? coarseDist[qi * nprobe + pi] : 0.0f;
                const float bias = coarse / (float)M;
                for (int i = 0; i < tableStride; ++i) {
                    tab[i] = pc[i] - 2.0f * sim2[(size_t)i] + bias;
                }
            }
            continue;
        }

        for (int pi = 0; pi < nprobe; ++pi) {
            idx_t listId = coarseAssign[qi * nprobe + pi];
            float* tab = tables + ((size_t)qi * nprobe + pi) * M * ksub;

            if (listId < 0) {
                float sentinel = isL2 ? 1e38f : -1e38f;
                std::fill_n(tab, tableStride, sentinel);
                continue;
            }

            // L2 tables depend on coarse centroid via residual query.
            if (coarseCentroids) {
                const float* cc = coarseCentroids + listId * d;
                for (int j = 0; j < d; ++j) {
                    residual[(size_t)j] = query[j] - cc[j];
                }
            } else {
                std::memcpy(residual.data(), query, (size_t)d * sizeof(float));
            }
            pq.compute_distance_table(residual.data(), tab);
        }
    }
}

// ============================================================
//  Train / add / reset
// ============================================================

void MetalIndexIVFPQ::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->train(n, x);
    if (cpuIndex_->metric_type == METRIC_L2 && cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    }
    is_trained = cpuIndex_->is_trained;
    if (is_trained) {
        uploadCentroids_();
        uploadPQCentroids_();
    }
}

void MetalIndexIVFPQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) return;

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    idx_t oldNt = cpuIndex_->ntotal;
    cpuIndex_->add(n, x);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        size_t code_size = cpuIndex_->pq.code_size;
        std::vector<uint8_t> encoded(n * code_size);
        cpuIndex_->pq.compute_codes(x, encoded.data(), n);

        std::vector<idx_t> ids(n);
        for (idx_t i = 0; i < n; ++i) ids[i] = oldNt + i;
        gpuIvf_->appendCodes(n, encoded.data(), list_nos.data(), ids.data());
    }
}

void MetalIndexIVFPQ::add_with_ids(
        idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(xids != nullptr);
    if (n == 0) return;

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    cpuIndex_->add_with_ids(n, x, xids);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        size_t code_size = cpuIndex_->pq.code_size;
        std::vector<uint8_t> encoded(n * code_size);
        cpuIndex_->pq.compute_codes(x, encoded.data(), n);
        gpuIvf_->appendCodes(n, encoded.data(), list_nos.data(), xids);
    }
}

void MetalIndexIVFPQ::reset() {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->reset();
    ntotal = 0;
    if (gpuIvf_) gpuIvf_->reset();
}

// ============================================================
//  Search
// ============================================================

void MetalIndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);

    const float inf = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const bool isL2 = (metric_type == METRIC_L2);

    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    id<MTLDevice> device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();

    if (!device || !queue || !gpuIvf_ || !gpuIvf_->codesBuffer() ||
        !gpuIvf_->idsBuffer() || !gpuIvf_->listOffsetGpuBuffer() ||
        !gpuIvf_->listLengthGpuBuffer()) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    const int maxK = getMetalDistanceMaxK();
    if (k > maxK) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    size_t nprobe = cpuIndex_->nprobe;
    if (auto* ivfParams = dynamic_cast<const IVFSearchParameters*>(params)) {
        if (ivfParams->nprobe > 0)
            nprobe = ivfParams->nprobe;
    }
    nprobe = std::min(nprobe, cpuIndex_->nlist);
    if (nprobe == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i] = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    // Coarse quantization on CPU.
    std::vector<float> coarseDistVec((size_t)n * nprobe);
    std::vector<idx_t> coarseAssignVec((size_t)n * nprobe);
    cpuIndex_->quantizer->search(
            n, x, (idx_t)nprobe, coarseDistVec.data(),
            coarseAssignVec.data());

    int M = (int)cpuIndex_->pq.M;
    int ksub = (int)cpuIndex_->pq.ksub;

    // Compute lookup tables on CPU directly into the upload buffer.
    size_t tableSize = (size_t)n * nprobe * M * ksub;
    size_t tableBytes = tableSize * sizeof(float);

    // Allocate GPU buffers.
    size_t outDistBytes = (size_t)n * (size_t)k * sizeof(float);
    size_t outIdxBytes = (size_t)n * (size_t)k * sizeof(int64_t);
    size_t queryBytes = (size_t)n * (size_t)d * sizeof(float);
    size_t perListBytes = (size_t)n * nprobe * (size_t)k * sizeof(float);
    size_t perListIdxB = (size_t)n * nprobe * (size_t)k * sizeof(int64_t);
    size_t coarseBytes = (size_t)n * nprobe * sizeof(int32_t);

    ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queryBytes);
    ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
    ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
    ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
    ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
    ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);
    ensureSearchBuf_(lookupTableBuf_, lookupTableCap_, tableBytes);

    if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
        !searchPerListDistBuf_ || !searchPerListIdxBuf_ ||
        !searchCoarseBuf_ || !lookupTableBuf_) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }
    std::memcpy([searchQueriesBuf_ contents], x, queryBytes);

    // Upload coarse assignments.
    auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
    for (size_t i = 0; i < (size_t)n * nprobe; ++i)
        coarseDst[i] = (int32_t)coarseAssignVec[i];
    int avgListLen = cpuIndex_->nlist > 0
            ? (int)(cpuIndex_->ntotal / cpuIndex_->nlist)
            : 0;
    bool lutBuiltOnGpu = false;
    bool ok = false;
    if (gpuIvf_->pqCentroidsBuffer()) {
        ok = runMetalIVFPQFullSearch(
                device,
                queue,
                searchQueriesBuf_,
                searchCoarseBuf_,
                centroidBuf_,
                gpuIvf_->pqCentroidsBuffer(),
                lookupTableBuf_,
                gpuIvf_->codesBuffer(),
                gpuIvf_->idsBuffer(),
                gpuIvf_->listOffsetGpuBuffer(),
                gpuIvf_->listLengthGpuBuffer(),
                (int)n,
                d,
                M,
                (int)k,
                (int)nprobe,
                avgListLen,
                isL2,
                searchOutDistBuf_,
                searchOutIdxBuf_,
                searchPerListDistBuf_,
                searchPerListIdxBuf_);
        lutBuiltOnGpu = ok;
    }
    if (!lutBuiltOnGpu) {
        auto* lookupDst = reinterpret_cast<float*>([lookupTableBuf_ contents]);
        computeLookupTables_(
                (int)n,
                x,
                (int)nprobe,
                coarseAssignVec.data(),
                coarseDistVec.data(),
                lookupDst);
        ok = runMetalIVFPQScan(
                device, queue,
                lookupTableBuf_,
                gpuIvf_->codesBuffer(),
                gpuIvf_->idsBuffer(),
                gpuIvf_->listOffsetGpuBuffer(),
                gpuIvf_->listLengthGpuBuffer(),
                searchCoarseBuf_,
                (int)n, M, (int)k, (int)nprobe, avgListLen, isL2,
                searchOutDistBuf_, searchOutIdxBuf_,
                searchPerListDistBuf_, searchPerListIdxBuf_);
    }

    if (!ok) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    const float* outDistPtr = reinterpret_cast<const float*>(
            [searchOutDistBuf_ contents]);
    const int64_t* outIdxPtr = reinterpret_cast<const int64_t*>(
            [searchOutIdxBuf_ contents]);
    for (idx_t qi = 0; qi < n; ++qi) {
        for (idx_t j = 0; j < k; ++j) {
            size_t pos = (size_t)qi * (size_t)k + (size_t)j;
            int64_t globalId = outIdxPtr[pos];
            labels[pos] = (globalId < 0) ? -1 : (idx_t)globalId;
            distances[pos] = outDistPtr[pos];
        }
    }
}

// ============================================================
//  Copy from / to
// ============================================================

void MetalIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT(src->pq.nbits == 8);
    FAISS_THROW_IF_NOT_FMT(
            src->nlist == cpuIndex_->nlist,
            "copyFrom: nlist mismatch (%zd vs %zd)",
            (size_t)src->nlist, (size_t)cpuIndex_->nlist);

    reset();

    if (!src->is_trained) {
        is_trained = false;
        return;
    }

    // Copy quantizer centroids (allow non-IndexFlat CPU coarse quantizers by
    // reconstructing centroid vectors).
    FAISS_THROW_IF_NOT_MSG(src->quantizer, "copyFrom: source quantizer is null");
    auto* ourQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer not IndexFlat");
    ourQ->reset();
    if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        ourQ->add(src->nlist, coarse.data());
    }

    // Copy PQ centroids.
    cpuIndex_->pq = src->pq;
    cpuIndex_->is_trained = true;
    cpuIndex_->nprobe = src->nprobe;
    cpuIndex_->by_residual = src->by_residual;
    cpuIndex_->use_precomputed_table = src->use_precomputed_table;
    if (cpuIndex_->metric_type == METRIC_L2 && cpuIndex_->by_residual) {
        cpuIndex_->precompute_table();
    }
    is_trained = true;

    // Gather IVF list data.
    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l)
        totalN += src->invlists->list_size(l);

    if (totalN > 0) {
        size_t code_size = src->pq.code_size;
        std::vector<uint8_t> allCodes(totalN * code_size);
        std::vector<idx_t> allListNos(totalN);
        std::vector<idx_t> allIds(totalN);
        size_t pos = 0;

        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0) continue;
            const uint8_t* codes = src->invlists->get_codes(l);
            const idx_t* ids = src->invlists->get_ids(l);

            cpuIndex_->invlists->add_entries(l, ls, ids, codes);

            std::memcpy(allCodes.data() + pos * code_size,
                        codes, ls * code_size);
            std::memcpy(allIds.data() + pos, ids, ls * sizeof(idx_t));
            for (size_t i = 0; i < ls; ++i)
                allListNos[pos + i] = (idx_t)l;
            pos += ls;
        }

        cpuIndex_->ntotal = (idx_t)totalN;
        ntotal = (idx_t)totalN;

        if (gpuIvf_)
            gpuIvf_->appendCodes(
                    (idx_t)totalN, allCodes.data(),
                    allListNos.data(), allIds.data());
    }

    uploadCentroids_();
    uploadPQCentroids_();
}

void MetalIndexIVFPQ::copyTo(faiss::IndexIVFPQ* dst) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(dst);

    auto* srcQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    auto* dstQ = dynamic_cast<faiss::IndexFlat*>(dst->quantizer);
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyTo: internal quantizer not IndexFlat");
    FAISS_THROW_IF_NOT_MSG(dstQ, "copyTo: destination quantizer not IndexFlat");

    dstQ->reset();
    if (srcQ->ntotal > 0)
        dstQ->add(srcQ->ntotal, srcQ->get_xb());

    dst->pq = cpuIndex_->pq;
    dst->metric_type = cpuIndex_->metric_type;
    dst->metric_arg = cpuIndex_->metric_arg;
    dst->d = cpuIndex_->d;
    dst->nlist = cpuIndex_->nlist;
    dst->nprobe = cpuIndex_->nprobe;
    dst->is_trained = cpuIndex_->is_trained;
    dst->by_residual = cpuIndex_->by_residual;
    dst->use_precomputed_table = 0;

    for (size_t l = 0; l < (size_t)cpuIndex_->nlist; ++l) {
        size_t ls = cpuIndex_->invlists->list_size(l);
        if (ls == 0) continue;
        const uint8_t* codes = cpuIndex_->invlists->get_codes(l);
        const idx_t* ids = cpuIndex_->invlists->get_ids(l);
        dst->invlists->add_entries(l, ls, ids, codes);
    }
    dst->ntotal = cpuIndex_->ntotal;
}

// ============================================================
//  Accessors
// ============================================================

void MetalIndexIVFPQ::updateQuantizer() {
    uploadCentroids_();
    uploadPQCentroids_();
}

std::vector<idx_t> MetalIndexIVFPQ::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0) return {};
    const idx_t* ids = cpuIndex_->invlists->get_ids(listId);
    return std::vector<idx_t>(ids, ids + ls);
}

void MetalIndexIVFPQ::reclaimMemory() {
    // No-op: Metal unified memory doesn't require explicit reclaim.
}

void MetalIndexIVFPQ::reserveMemory(idx_t numVecs) {
    if (gpuIvf_) {
        gpuIvf_->reserveMemory(numVecs);
    }
}

idx_t MetalIndexIVFPQ::nlist() const {
    return cpuIndex_ ? cpuIndex_->nlist : 0;
}

size_t MetalIndexIVFPQ::nprobe() const {
    return cpuIndex_ ? cpuIndex_->nprobe : 1;
}

int MetalIndexIVFPQ::getNumSubQuantizers() const {
    return cpuIndex_ ? (int)cpuIndex_->pq.M : 0;
}

} // namespace gpu_metal
} // namespace faiss
