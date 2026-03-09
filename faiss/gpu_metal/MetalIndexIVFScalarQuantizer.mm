// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndexIVFScalarQuantizer.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/impl/MetalIVFSQ.h>

#include <cstring>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu_metal {

static MetalSQType toMetalSQType(faiss::ScalarQuantizer::QuantizerType qt) {
    switch (qt) {
        case faiss::ScalarQuantizer::QT_8bit:
            return MetalSQType::SQ8;
        case faiss::ScalarQuantizer::QT_fp16:
            return MetalSQType::FP16;
        default:
            FAISS_THROW_FMT(
                    "Metal IVFSQ: unsupported quantizer type %d "
                    "(only QT_8bit and QT_fp16 supported)",
                    (int)qt);
    }
}

// ============================================================
//  Constructors / destructor
// ============================================================

MetalIndexIVFScalarQuantizer::MetalIndexIVFScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        int dims,
        idx_t nlist,
        faiss::ScalarQuantizer::QuantizerType sqType,
        faiss::MetricType metric,
        bool byResidual,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(resources, dims, metric, metricArg, config) {
    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
 
    cpuIndex_ = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            quantizer, (size_t)d, (size_t)nlist, sqType, metric, false);
    cpuIndex_->own_fields = true;

    MetalSQType mst = toMetalSQType(sqType);
    gpuIvf_ = std::make_unique<MetalIVFSQImpl>(
            resources, dims, nlist, mst, metric, metricArg);
}

MetalIndexIVFScalarQuantizer::MetalIndexIVFScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexIVFScalarQuantizer* cpuIndex,
        MetalIndexConfig config)
        : MetalIndex(
                  resources,
                  (int)cpuIndex->d,
                  cpuIndex->metric_type,
                  cpuIndex->metric_arg,
                  config) {
    MetalSQType mst = toMetalSQType(cpuIndex->sq.qtype);

    faiss::IndexFlat* quantizer =
            (cpuIndex->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)cpuIndex->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)cpuIndex->d);
    cpuIndex_ = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            quantizer,
            cpuIndex->d,
            cpuIndex->nlist,
            cpuIndex->sq.qtype,
            cpuIndex->metric_type,
            /*by_residual=*/false);
    cpuIndex_->own_fields = true;

    gpuIvf_ = std::make_unique<MetalIVFSQImpl>(
            resources,
            (int)cpuIndex->d,
            cpuIndex->nlist,
            mst,
            cpuIndex->metric_type,
            cpuIndex->metric_arg);

    copyFrom(cpuIndex);
}

MetalIndexIVFScalarQuantizer::~MetalIndexIVFScalarQuantizer() = default;

// ============================================================
//  Internal helpers
// ============================================================

void MetalIndexIVFScalarQuantizer::ensureSearchBuf_(
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

void MetalIndexIVFScalarQuantizer::uploadCentroids_() const {
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

    if (centroidBuf_ && metric_type == METRIC_L2) {
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

void MetalIndexIVFScalarQuantizer::uploadSQTables_() const {
    if (!cpuIndex_ || !gpuIvf_) return;
    if (gpuIvf_->sqType() != MetalSQType::SQ8) return;

    const auto& trained = cpuIndex_->sq.trained;
    if (trained.size() < (size_t)d * 2) return;
    gpuIvf_->setSQTables(trained.data());
}

// ============================================================
//  Index operations
// ============================================================

void MetalIndexIVFScalarQuantizer::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->train(n, x);
    is_trained = cpuIndex_->is_trained;
    if (is_trained) {
        uploadCentroids_();
        uploadSQTables_();
    }
}

void MetalIndexIVFScalarQuantizer::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) return;

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    idx_t oldNt = cpuIndex_->ntotal;
    cpuIndex_->add(n, x);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        size_t codeSize = gpuIvf_->codeSize();
        std::vector<uint8_t> encoded(n * codeSize);
        cpuIndex_->encode_vectors(n, x, list_nos.data(), encoded.data());

        std::vector<idx_t> ids(n);
        for (idx_t i = 0; i < n; ++i) {
            ids[i] = oldNt + i;
        }
        gpuIvf_->appendCodes(n, encoded.data(), list_nos.data(), ids.data());
    }
}

void MetalIndexIVFScalarQuantizer::add_with_ids(
        idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    if (n == 0) return;
    FAISS_THROW_IF_NOT(xids != nullptr);

    std::vector<idx_t> list_nos(n);
    cpuIndex_->quantizer->assign(n, x, list_nos.data());

    cpuIndex_->add_with_ids(n, x, xids);
    ntotal = cpuIndex_->ntotal;

    if (gpuIvf_) {
        size_t codeSize = gpuIvf_->codeSize();
        std::vector<uint8_t> encoded(n * codeSize);
        cpuIndex_->encode_vectors(n, x, list_nos.data(), encoded.data());
        gpuIvf_->appendCodes(n, encoded.data(), list_nos.data(), xids);
    }
}

void MetalIndexIVFScalarQuantizer::reset() {
    FAISS_THROW_IF_NOT(cpuIndex_);
    cpuIndex_->reset();
    ntotal = 0;
    if (gpuIvf_) gpuIvf_->reset();
}

void MetalIndexIVFScalarQuantizer::search(
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
    const bool isL2 = (metric_type == METRIC_L2);

    if (cpuIndex_->ntotal == 0 || n == 0) {
        for (idx_t i = 0; i < n * k; ++i) {
            labels[i]    = -1;
            distances[i] = isL2 ? inf : negInf;
        }
        return;
    }

    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

    id<MTLDevice>       device = resources_->getDevice();
    id<MTLCommandQueue> queue  = resources_->getCommandQueue();

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

    // Coarse quantization on CPU.
    std::vector<float>  coarseDistVec((size_t)n * nprobe);
    std::vector<idx_t>  coarseAssignVec((size_t)n * nprobe);
    cpuIndex_->quantizer->search(
            n, x, (idx_t)nprobe, coarseDistVec.data(),
            coarseAssignVec.data());

    size_t queriesBytes   = (size_t)n * (size_t)d * sizeof(float);
    size_t outDistBytes   = (size_t)n * (size_t)k * sizeof(float);
    size_t outIdxBytes    = (size_t)n * (size_t)k * sizeof(int64_t);
    size_t perListBytes   = (size_t)n * nprobe * (size_t)k * sizeof(float);
    size_t perListIdxB    = (size_t)n * nprobe * (size_t)k * sizeof(int64_t);
    size_t coarseBytes    = (size_t)n * nprobe * sizeof(int32_t);

    ensureSearchBuf_(searchQueriesBuf_,     searchQueriesCap_,     queriesBytes);
    ensureSearchBuf_(searchOutDistBuf_,     searchOutDistCap_,     outDistBytes);
    ensureSearchBuf_(searchOutIdxBuf_,      searchOutIdxCap_,      outIdxBytes);
    ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
    ensureSearchBuf_(searchPerListIdxBuf_,  searchPerListIdxCap_,  perListIdxB);
    ensureSearchBuf_(searchCoarseBuf_,      searchCoarseCap_,      coarseBytes);

    if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
        !searchPerListDistBuf_ || !searchPerListIdxBuf_ || !searchCoarseBuf_) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    std::memcpy([searchQueriesBuf_ contents], x, queriesBytes);
    auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
    for (size_t i = 0; i < (size_t)n * nprobe; ++i) {
        coarseDst[i] = (int32_t)coarseAssignVec[i];
    }

    MetalSQType mst = gpuIvf_->sqType();
    bool ok = runMetalIVFSQScan(
            device, queue,
            searchQueriesBuf_,
            gpuIvf_->codesBuffer(),
            gpuIvf_->idsBuffer(),
            gpuIvf_->listOffsetGpuBuffer(),
            gpuIvf_->listLengthGpuBuffer(),
            searchCoarseBuf_,
            (int)n, d, (int)k, (int)nprobe, isL2,
            mst,
            gpuIvf_->sqTablesBuffer(),
            searchOutDistBuf_, searchOutIdxBuf_,
            searchPerListDistBuf_, searchPerListIdxBuf_);

    if (!ok) {
        cpuIndex_->search(n, x, k, distances, labels);
        return;
    }

    const float*   outDistPtr = reinterpret_cast<const float*>  ([searchOutDistBuf_ contents]);
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

// ============================================================
//  Copy from / to
// ============================================================

void MetalIndexIVFScalarQuantizer::copyFrom(
        const faiss::IndexIVFScalarQuantizer* src) {
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(src);
    FAISS_THROW_IF_NOT_FMT(
            src->nlist == cpuIndex_->nlist,
            "copyFrom: nlist mismatch (%zd vs %zd)",
            (size_t)src->nlist, (size_t)cpuIndex_->nlist);
    reset();

    if (!src->is_trained) {
        is_trained = false;
        return;
    }

    auto* srcQ = dynamic_cast<const faiss::IndexFlat*>(src->quantizer);
    auto* ourQ = dynamic_cast<faiss::IndexFlat*>(cpuIndex_->quantizer);
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyFrom: source quantizer is not IndexFlat");
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer is not IndexFlat");
    ourQ->reset();
    if (srcQ->ntotal > 0) {
        ourQ->add(srcQ->ntotal, srcQ->get_xb());
    }

    cpuIndex_->sq = src->sq;
    cpuIndex_->by_residual = false;
    cpuIndex_->is_trained = true;
    cpuIndex_->nprobe = src->nprobe;
    is_trained = true;

    size_t codeSize = gpuIvf_ ? gpuIvf_->codeSize() : (size_t)src->sq.code_size;
    bool needReencode = src->by_residual;

    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l) {
        totalN += src->invlists->list_size(l);
    }

    if (totalN > 0) {
        std::vector<uint8_t> allCodes(totalN * codeSize);
        std::vector<idx_t>   allListNos(totalN);
        std::vector<idx_t>   allIds(totalN);
        size_t pos = 0;

        if (needReencode) {
            // Source stores residual-encoded codes. Decode to float vectors,
            // then re-encode with by_residual=false for GPU-friendly storage.
            for (size_t l = 0; l < (size_t)src->nlist; ++l) {
                size_t ls = src->invlists->list_size(l);
                if (ls == 0) continue;

                const uint8_t* srcCodes = src->invlists->get_codes(l);
                const idx_t* ids = src->invlists->get_ids(l);

                std::vector<idx_t> listNos(ls, (idx_t)l);
                std::vector<float> decoded(ls * (size_t)d);
                src->decode_vectors(ls, srcCodes, listNos.data(), decoded.data());

                cpuIndex_->encode_vectors(
                        ls, decoded.data(), listNos.data(),
                        allCodes.data() + pos * codeSize);

                std::memcpy(allIds.data() + pos, ids, ls * sizeof(idx_t));
                for (size_t i = 0; i < ls; ++i) {
                    allListNos[pos + i] = (idx_t)l;
                }
                pos += ls;
            }
        } else {
            for (size_t l = 0; l < (size_t)src->nlist; ++l) {
                size_t ls = src->invlists->list_size(l);
                if (ls == 0) continue;

                const uint8_t* codes = src->invlists->get_codes(l);
                const idx_t* ids = src->invlists->get_ids(l);

                std::memcpy(allCodes.data() + pos * codeSize, codes, ls * codeSize);
                std::memcpy(allIds.data() + pos, ids, ls * sizeof(idx_t));
                for (size_t i = 0; i < ls; ++i) {
                    allListNos[pos + i] = (idx_t)l;
                }
                pos += ls;
            }
        }

        // Add re-encoded (or direct) codes to our CPU index's inverted lists.
        pos = 0;
        for (size_t l = 0; l < (size_t)src->nlist; ++l) {
            size_t ls = src->invlists->list_size(l);
            if (ls == 0) continue;
            cpuIndex_->invlists->add_entries(
                    l, ls,
                    allIds.data() + pos,
                    allCodes.data() + pos * codeSize);
            pos += ls;
        }

        cpuIndex_->ntotal = (idx_t)totalN;
        ntotal = (idx_t)totalN;

        if (gpuIvf_) {
            gpuIvf_->appendCodes(
                    (idx_t)totalN,
                    allCodes.data(),
                    allListNos.data(),
                    allIds.data());
        }
    }

    uploadCentroids_();
    uploadSQTables_();
}

void MetalIndexIVFScalarQuantizer::copyTo(
        faiss::IndexIVFScalarQuantizer* dst) const {
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
    dst->metric_arg  = cpuIndex_->metric_arg;
    dst->d           = cpuIndex_->d;
    dst->nlist       = cpuIndex_->nlist;
    dst->nprobe      = cpuIndex_->nprobe;
    dst->is_trained  = cpuIndex_->is_trained;
    dst->sq          = cpuIndex_->sq;
    dst->by_residual = cpuIndex_->by_residual;

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
//  Reconstruct / accessors
// ============================================================

void MetalIndexIVFScalarQuantizer::reconstruct(
        idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_,
                           "MetalIndexIVFScalarQuantizer: no internal index");
    cpuIndex_->reconstruct(key, recons);
}

void MetalIndexIVFScalarQuantizer::reconstruct_n(
        idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_,
                           "MetalIndexIVFScalarQuantizer: no internal index");
    cpuIndex_->reconstruct_n(i0, ni, recons);
}

idx_t MetalIndexIVFScalarQuantizer::nlist() const {
    return cpuIndex_ ? cpuIndex_->nlist : 0;
}

size_t MetalIndexIVFScalarQuantizer::nprobe() const {
    return cpuIndex_ ? cpuIndex_->nprobe : 1;
}

faiss::ScalarQuantizer::QuantizerType
MetalIndexIVFScalarQuantizer::sqQuantizerType() const {
    return cpuIndex_ ? cpuIndex_->sq.qtype
                     : faiss::ScalarQuantizer::QT_8bit;
}

} // namespace gpu_metal
} // namespace faiss
