// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFFlat.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/impl/MetalIVFFlat.h>

#include <cstring>
#include <limits>
#include <vector>

namespace {
void floatToHalf(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        __fp16 h = (__fp16)src[i];
        std::memcpy(&dst[i], &h, sizeof(uint16_t));
    }
}
} // namespace

namespace faiss {
namespace gpu_metal {

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config) {
    // Simple CPU quantizer: IndexFlatL2 or IndexFlatIP
    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, (size_t)d, (size_t)nlist, metric);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources, (int)d, nlist, metric, metricArg);
}

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexIVFFlat* cpuIndex,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  (int)cpuIndex->d,
                  cpuIndex->metric_type,
                  cpuIndex->metric_arg,
                  config) {
    faiss::IndexFlat* quantizer = (cpuIndex->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)cpuIndex->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)cpuIndex->d);
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, cpuIndex->d, cpuIndex->nlist, cpuIndex->metric_type);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)cpuIndex->d,
            cpuIndex->nlist,
            cpuIndex->metric_type,
            cpuIndex->metric_arg);
    copyFrom(cpuIndex);
}

MetalIndexIVFFlat::~MetalIndexIVFFlat() = default;

void MetalIndexIVFFlat::uploadCentroids_() const {
    if (!cpuIndex_ || !cpuIndex_->quantizer || !resources_) {
        return;
    }
    auto* flatQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    if (!flatQ || flatQ->ntotal == 0) {
        centroidBuf_ = nil;
        centroidNormsBuf_ = nil;
        return;
    }
    size_t nCentroids = (size_t)flatQ->ntotal;
    const bool fp16 = config_.useFloat16CoarseQuantizer;
    size_t elemSize = fp16 ? sizeof(uint16_t) : sizeof(float);
    size_t bytes = nCentroids * (size_t)d * elemSize;
    id<MTLDevice> device = resources_->getDevice();
    if (!device) {
        return;
    }
    centroidBuf_ = [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    if (centroidBuf_) {
        const float* src = flatQ->get_xb();
        if (fp16) {
            floatToHalf(src,
                        reinterpret_cast<uint16_t*>([centroidBuf_ contents]),
                        nCentroids * (size_t)d);
        } else {
            std::memcpy([centroidBuf_ contents], src, bytes);
        }
    }

    // Pre-compute centroid L2 norms on GPU (float32 centroids only).
    if (centroidBuf_ && metric_type == METRIC_L2 && !fp16) {
        size_t normBytes = nCentroids * sizeof(float);
        centroidNormsBuf_ = [device newBufferWithLength:normBytes
                                               options:MTLResourceStorageModeShared];
        if (centroidNormsBuf_) {
            id<MTLCommandQueue> queue = resources_->getCommandQueue();
            if (!runMetalComputeNorms(device, queue, centroidBuf_,
                                      (int)nCentroids, d, centroidNormsBuf_)) {
                centroidNormsBuf_ = nil;
            }
        }
    } else {
        centroidNormsBuf_ = nil;
    }
}

void MetalIndexIVFFlat::ensureSearchBuf_(
        id<MTLBuffer>& buf,
        size_t& cap,
        size_t needed) const {
    if (buf != nil && cap >= needed) {
        return; // already large enough
    }
    // Grow by 2× to amortise future reallocations.
    size_t newCap = std::max(needed, cap * 2);
    id<MTLDevice> device = resources_->getDevice();
    buf = [device newBufferWithLength:newCap
                              options:MTLResourceStorageModeShared];
    cap = buf ? newCap : 0;
}

void MetalIndexIVFFlat::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->train(n, x);
    is_trained = cpuIndex_->is_trained;
    if (is_trained) {
        uploadCentroids_();
    }
}

void MetalIndexIVFFlat::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) {
        return;
    }
    // Compute list assignments for this batch (mirrors IndexIVF::add path).
    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    idx_t oldNt = cpuIndex_->ntotal;
    cpuIndex_->add(n, x);
    ntotal = cpuIndex_->ntotal;

    // Mirror IVF data into Metal IVF storage.
    if (gpuIvf_) {
        std::vector<idx_t> ids(n);
        for (idx_t i = 0; i < n; ++i) {
            ids[i] = oldNt + i;
        }
        gpuIvf_->appendVectors(n, x, list_nos.data(), ids.data());
    }
}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(xids != nullptr);

    // Compute list assignments for this batch.
    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    cpuIndex_->add_with_ids(n, x, xids);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        gpuIvf_->appendVectors(n, x, list_nos.data(), xids);
    }
}

void MetalIndexIVFFlat::reset() {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->reset();
    ntotal = 0;
    if (gpuIvf_) {
        gpuIvf_->reset();
    }
}

void MetalIndexIVFFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);

    const float inf    = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();

    // Empty index: mirror CPU IndexIVF behavior (labels = -1).
    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = (metric_type == METRIC_L2) ? inf : negInf;
        }
        return;
    }

    // Only L2 and IP are supported.
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);
    const bool isL2 = (metric_type == METRIC_L2);

    id<MTLDevice>      device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();

    // Fall back to CPU if Metal is not available or GPU IVF storage not ready.
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

    // Determine nprobe from params or index.
    size_t nprobe = cpuIndex_->nprobe;
    if (auto* ivfParams = dynamic_cast<const IVFSearchParameters*>(params)) {
        if (ivfParams->nprobe > 0) {
            nprobe = ivfParams->nprobe;
        }
    }
    nprobe = std::min(nprobe, cpuIndex_->nlist);
    if (nprobe == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    // ---- Upload centroid buffer if needed -----------------------------------
    if (!centroidBuf_) {
        uploadCentroids_();
    }

    int nlist = (int)cpuIndex_->nlist;

    // ---- Grow persistent search buffers ------------------------------------
    size_t queriesBytes    = (size_t)n * (size_t)d * sizeof(float);
    size_t outDistBytes    = (size_t)n * (size_t)k * sizeof(float);
    size_t outIdxBytes     = (size_t)n * (size_t)k * sizeof(int64_t);
    size_t perListBytes    = (size_t)n * nprobe * (size_t)k * sizeof(float);
    size_t perListIdxB     = (size_t)n * nprobe * (size_t)k * sizeof(int64_t);
    size_t coarseDistBytes = (size_t)n * nprobe * sizeof(float);
    size_t coarseIdxBytes  = (size_t)n * nprobe * sizeof(int32_t);
    size_t distMatBytes    = (size_t)n * (size_t)nlist * sizeof(float);

    ensureSearchBuf_(searchQueriesBuf_,      searchQueriesCap_,      queriesBytes);
    ensureSearchBuf_(searchOutDistBuf_,      searchOutDistCap_,      outDistBytes);
    ensureSearchBuf_(searchOutIdxBuf_,       searchOutIdxCap_,       outIdxBytes);
    ensureSearchBuf_(searchPerListDistBuf_,  searchPerListDistCap_,  perListBytes);
    ensureSearchBuf_(searchPerListIdxBuf_,   searchPerListIdxCap_,   perListIdxB);
    ensureSearchBuf_(coarseOutDistBuf_,      coarseOutDistCap_,      coarseDistBytes);
    ensureSearchBuf_(coarseOutIdxBuf_,       coarseOutIdxCap_,       coarseIdxBytes);
    ensureSearchBuf_(distMatrixBuf_,         distMatrixCap_,         distMatBytes);

    if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
        !searchPerListDistBuf_ || !searchPerListIdxBuf_ ||
        !coarseOutDistBuf_ || !coarseOutIdxBuf_ || !distMatrixBuf_) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    std::memcpy([searchQueriesBuf_ contents], x, queriesBytes);

    // ---- Single command buffer: coarse quant + IVF scan + merge ----------
    bool ok = false;
    if (centroidBuf_ && nprobe <= (size_t)getMetalDistanceMaxK()) {
        int avgListLen = (ntotal > 0 && nlist > 0)
                       ? (int)(ntotal / nlist) : 256;
        ok = runMetalIVFFlatFullSearch(
                device, queue,
                searchQueriesBuf_,
                (int)n, d, (int)k, (int)nprobe, isL2,
                centroidBuf_, nlist,
                gpuIvf_->codesBuffer(),
                gpuIvf_->idsBuffer(),
                gpuIvf_->listOffsetGpuBuffer(),
                gpuIvf_->listLengthGpuBuffer(),
                searchOutDistBuf_,
                searchOutIdxBuf_,
                searchPerListDistBuf_,
                searchPerListIdxBuf_,
                coarseOutDistBuf_,
                coarseOutIdxBuf_,
                distMatrixBuf_,
                centroidNormsBuf_,
                avgListLen,
                gpuIvf_->interleavedCodesBuffer(),
                gpuIvf_->interleavedCodesOffsetBuffer(),
                config_.useFloat16CoarseQuantizer);
    }

    if (!ok) {
        // Fallback: CPU coarse + GPU IVF scan (two command buffers).
        std::vector<float>  coarseDistVec((size_t)n * nprobe);
        std::vector<idx_t>  coarseAssignVec((size_t)n * nprobe);
        cpuIndex_->quantizer->search(
                n, x, (idx_t)nprobe, coarseDistVec.data(),
                coarseAssignVec.data());
        size_t coarseBytes = (size_t)n * nprobe * sizeof(int32_t);
        ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);
        if (!searchCoarseBuf_) {
            cpuIndex_->search(n, x, k, distances, labels);
            return;
        }
        auto* dst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
        for (size_t i = 0; i < (size_t)n * nprobe; ++i) {
            dst[i] = (int32_t)coarseAssignVec[i];
        }
        ok = runMetalIVFFlatScan(
                device, queue,
                searchQueriesBuf_,
                gpuIvf_->codesBuffer(),
                gpuIvf_->idsBuffer(),
                gpuIvf_->listOffsetGpuBuffer(),
                gpuIvf_->listLengthGpuBuffer(),
                searchCoarseBuf_,
                (int)n, d, (int)k, (int)nprobe, isL2,
                searchOutDistBuf_, searchOutIdxBuf_,
                searchPerListDistBuf_, searchPerListIdxBuf_,
                gpuIvf_->interleavedCodesBuffer(),
                gpuIvf_->interleavedCodesOffsetBuffer());
    }

    if (!ok) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    // ---- Copy results back -----------------------------------------------
    const float*   outDistPtr = reinterpret_cast<const float*  >([searchOutDistBuf_ contents]);
    const int64_t* outIdxPtr  = reinterpret_cast<const int64_t*>([searchOutIdxBuf_  contents]);

    for (idx_t qi = 0; qi < n; ++qi) {
        for (idx_t j = 0; j < k; ++j) {
            size_t pos        = (size_t)qi * (size_t)k + (size_t)j;
            int64_t globalId  = outIdxPtr[pos];
            labels   [pos]    = (globalId < 0) ? -1 : (idx_t)globalId;
            distances[pos]    = outDistPtr[pos];
        }
    }
}

void MetalIndexIVFFlat::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    (void)centroid_dis;
    (void)store_pairs;
    (void)stats;

    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(assign);

    const float inf    = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const bool isL2 = (metric_type == METRIC_L2);

    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    size_t nprobe = cpuIndex_->nprobe;
    if (params && params->nprobe > 0) {
        nprobe = params->nprobe;
    }
    nprobe = std::min(nprobe, cpuIndex_->nlist);

    id<MTLDevice>       device = resources_->getDevice();
    id<MTLCommandQueue> queue  = resources_->getCommandQueue();

    const int maxK = getMetalDistanceMaxK();
    if (!device || !queue || !gpuIvf_ || !gpuIvf_->codesBuffer() ||
        !gpuIvf_->idsBuffer() || !gpuIvf_->listOffsetGpuBuffer() ||
        !gpuIvf_->listLengthGpuBuffer() || k > maxK) {
        cpuIndex_->search_preassigned(
                n, x, k, assign, centroid_dis,
                distances, labels, store_pairs, params, stats);
        return;
    }

    size_t queriesBytes    = (size_t)n * (size_t)d * sizeof(float);
    size_t outDistBytes    = (size_t)n * (size_t)k * sizeof(float);
    size_t outIdxBytes     = (size_t)n * (size_t)k * sizeof(int64_t);
    size_t perListBytes    = (size_t)n * nprobe * (size_t)k * sizeof(float);
    size_t perListIdxB     = (size_t)n * nprobe * (size_t)k * sizeof(int64_t);
    size_t coarseBytes     = (size_t)n * nprobe * sizeof(int32_t);

    ensureSearchBuf_(searchQueriesBuf_,     searchQueriesCap_,     queriesBytes);
    ensureSearchBuf_(searchOutDistBuf_,     searchOutDistCap_,     outDistBytes);
    ensureSearchBuf_(searchOutIdxBuf_,      searchOutIdxCap_,      outIdxBytes);
    ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
    ensureSearchBuf_(searchPerListIdxBuf_,  searchPerListIdxCap_,  perListIdxB);
    ensureSearchBuf_(searchCoarseBuf_,      searchCoarseCap_,      coarseBytes);

    if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
        !searchPerListDistBuf_ || !searchPerListIdxBuf_ || !searchCoarseBuf_) {
        cpuIndex_->search_preassigned(
                n, x, k, assign, centroid_dis,
                distances, labels, store_pairs, params, stats);
        return;
    }

    std::memcpy([searchQueriesBuf_ contents], x, queriesBytes);

    auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
    for (size_t i = 0; i < (size_t)n * nprobe; ++i) {
        coarseDst[i] = (int32_t)assign[i];
    }

    bool ok = runMetalIVFFlatScan(
            device, queue,
            searchQueriesBuf_,
            gpuIvf_->codesBuffer(),
            gpuIvf_->idsBuffer(),
            gpuIvf_->listOffsetGpuBuffer(),
            gpuIvf_->listLengthGpuBuffer(),
            searchCoarseBuf_,
            (int)n, d, (int)k, (int)nprobe, isL2,
            searchOutDistBuf_, searchOutIdxBuf_,
            searchPerListDistBuf_, searchPerListIdxBuf_,
            gpuIvf_->interleavedCodesBuffer(),
            gpuIvf_->interleavedCodesOffsetBuffer());

    if (!ok) {
        cpuIndex_->search_preassigned(
                n, x, k, assign, centroid_dis,
                distances, labels, store_pairs, params, stats);
        return;
    }

    const float*   outDistPtr = reinterpret_cast<const float*  >([searchOutDistBuf_ contents]);
    const int64_t* outIdxPtr  = reinterpret_cast<const int64_t*>([searchOutIdxBuf_  contents]);
    for (idx_t qi = 0; qi < n; ++qi) {
        for (idx_t j = 0; j < k; ++j) {
            size_t pos       = (size_t)qi * (size_t)k + (size_t)j;
            int64_t globalId = outIdxPtr[pos];
            labels   [pos]   = (globalId < 0) ? -1 : (idx_t)globalId;
            distances[pos]   = outDistPtr[pos];
        }
    }
}

void MetalIndexIVFFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "MetalIndexIVFFlat: no internal index");
    cpuIndex_->reconstruct(key, recons);
}

void MetalIndexIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "MetalIndexIVFFlat: no internal index");
    cpuIndex_->reconstruct_n(i0, ni, recons);
}

void MetalIndexIVFFlat::updateQuantizer() {
    uploadCentroids_();
}

std::vector<idx_t> MetalIndexIVFFlat::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0) return {};
    const idx_t* ids = cpuIndex_->invlists->get_ids(listId);
    return std::vector<idx_t>(ids, ids + ls);
}

std::vector<float> MetalIndexIVFFlat::getListVectorData(idx_t listId) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(listId >= 0 && listId < cpuIndex_->nlist);
    size_t ls = cpuIndex_->invlists->list_size(listId);
    if (ls == 0) return {};
    const uint8_t* codes = cpuIndex_->invlists->get_codes(listId);
    size_t floatCount = ls * (size_t)d;
    const float* fptr = reinterpret_cast<const float*>(codes);
    return std::vector<float>(fptr, fptr + floatCount);
}

void MetalIndexIVFFlat::reclaimMemory() {
    // No-op for now: Metal unified memory doesn't benefit from explicit
    // reclaim in the same way as discrete CUDA GPU memory.
}

void MetalIndexIVFFlat::reserveMemory(idx_t numVecs) {
    if (gpuIvf_) {
        gpuIvf_->reserveMemory(numVecs);
    }
}

idx_t MetalIndexIVFFlat::nlist() const {
    return cpuIndex_ ? cpuIndex_->nlist : 0;
}

size_t MetalIndexIVFFlat::nprobe() const {
    return cpuIndex_ ? cpuIndex_->nprobe : 1;
}

void MetalIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT_FMT(
            src->nlist == cpuIndex_->nlist,
            "copyFrom: nlist mismatch (%zd vs %zd)",
            (size_t)src->nlist,
            (size_t)cpuIndex_->nlist);
    reset();

    if (!src->is_trained) {
        is_trained = false;
        return;
    }

    // Copy quantizer centroids.
    auto* srcQ = dynamic_cast<const faiss::IndexFlat*>(src->quantizer);
    auto* ourQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyFrom: source quantizer is not IndexFlat");
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer is not IndexFlat");
    ourQ->reset();
    if (srcQ->ntotal > 0) {
        ourQ->add(srcQ->ntotal, srcQ->get_xb());
    }
    cpuIndex_->is_trained = true;
    cpuIndex_->nprobe = src->nprobe;
    is_trained = true;

    // Gather all vectors from inverted lists for a single GPU upload.
    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l) {
        totalN += src->invlists->list_size(l);
    }

    if (totalN > 0) {
        std::vector<float> allCodes(totalN * (size_t)d);
        std::vector<idx_t> allListNos(totalN);
        std::vector<idx_t> allIds(totalN);
        size_t pos = 0;

        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0) {
                continue;
            }
            const uint8_t* codes = src->invlists->get_codes(l);
            const idx_t* ids = src->invlists->get_ids(l);

            cpuIndex_->invlists->add_entries(l, ls, ids, codes);

            std::memcpy(
                    allCodes.data() + pos * (size_t)d,
                    codes,
                    ls * (size_t)d * sizeof(float));
            std::memcpy(allIds.data() + pos, ids, ls * sizeof(idx_t));
            for (size_t i = 0; i < ls; ++i) {
                allListNos[pos + i] = (idx_t)l;
            }
            pos += ls;
        }

        cpuIndex_->ntotal = (idx_t)totalN;
        ntotal = (idx_t)totalN;

        if (gpuIvf_) {
            gpuIvf_->appendVectors(
                    (idx_t)totalN,
                    allCodes.data(),
                    allListNos.data(),
                    allIds.data());
        }
    }

    uploadCentroids_();
}

void MetalIndexIVFFlat::copyTo(faiss::IndexIVFFlat* dst) const {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(dst);

    auto* srcQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    auto* dstQ = dynamic_cast<faiss::IndexFlat*>(dst->quantizer);
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyTo: internal quantizer is not IndexFlat");
    FAISS_THROW_IF_NOT_MSG(dstQ, "copyTo: destination quantizer is not IndexFlat");

    dstQ->reset();
    if (srcQ->ntotal > 0) {
        dstQ->add(srcQ->ntotal, srcQ->get_xb());
    }

    dst->metric_type = cpuIndex_->metric_type;
    dst->metric_arg = cpuIndex_->metric_arg;
    dst->d = cpuIndex_->d;
    dst->nlist = cpuIndex_->nlist;
    dst->nprobe = cpuIndex_->nprobe;
    dst->is_trained = cpuIndex_->is_trained;

    for (size_t l = 0; l < (size_t)cpuIndex_->nlist; ++l) {
        size_t ls = cpuIndex_->invlists->list_size(l);
        if (ls == 0) {
            continue;
        }
        const uint8_t* codes = cpuIndex_->invlists->get_codes(l);
        const idx_t* ids = cpuIndex_->invlists->get_ids(l);
        dst->invlists->add_entries(l, ls, ids, codes);
    }
    dst->ntotal = cpuIndex_->ntotal;
}

} // namespace gpu_metal
} // namespace faiss

