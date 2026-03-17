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
#include <faiss/invlists/DirectMap.h>

#include <cstring>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
constexpr size_t kDefaultIvfQueryTileBudgetBytes = 256ULL * 1024 * 1024;
constexpr size_t kMinIvfQueryTileBudgetBytes = 16ULL * 1024 * 1024;
constexpr size_t kMaxIvfQueryTileBudgetBytes = 4ULL * 1024 * 1024 * 1024;
constexpr faiss::idx_t kIVFFlatSupportedMaxK = 1024;
constexpr faiss::idx_t kAutoReserveMinBatch = 0;

size_t getIvfQueryTileBudgetBytes() {
    const char* envBytes = std::getenv("FAISS_METAL_IVF_QUERY_TILE_BYTES");
    if (envBytes && envBytes[0] != '\0') {
        char* end = nullptr;
        unsigned long long val = std::strtoull(envBytes, &end, 10);
        if (end != envBytes && val != 0) {
            size_t out = static_cast<size_t>(val);
            out = std::max(out, kMinIvfQueryTileBudgetBytes);
            out = std::min(out, kMaxIvfQueryTileBudgetBytes);
            return out;
        }
    }
    return kDefaultIvfQueryTileBudgetBytes;
}

faiss::idx_t getIvfAutoReserveMinBatch() {
    const char* env = std::getenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH");
    if (!env || env[0] == '\0') {
        return kAutoReserveMinBatch;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env) {
        return kAutoReserveMinBatch;
    }
    if (v > (unsigned long long)std::numeric_limits<faiss::idx_t>::max()) {
        return std::numeric_limits<faiss::idx_t>::max();
    }
    return (faiss::idx_t)v;
}

size_t chooseIvfSearchTileRows(
        size_t nq,
        int d,
        faiss::idx_t k,
        size_t nprobe,
        int nlist) {
    size_t perQuery = 0;
    perQuery += (size_t)d * sizeof(float);
    perQuery += (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (sizeof(float) + sizeof(int32_t));
    perQuery += (size_t)nlist * sizeof(float);
    if (perQuery == 0) {
        return nq;
    }

    size_t tile = getIvfQueryTileBudgetBytes() / perQuery;
    tile = std::max<size_t>(tile, 1);
    tile = std::min(tile, nq);
    return tile;
}

size_t chooseIvfPreassignedTileRows(
        size_t nq,
        int d,
        faiss::idx_t k,
        size_t nprobe) {
    size_t perQuery = 0;
    perQuery += (size_t)d * sizeof(float);
    perQuery += (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * sizeof(int32_t);
    if (perQuery == 0) {
        return nq;
    }

    size_t tile = getIvfQueryTileBudgetBytes() / perQuery;
    tile = std::max<size_t>(tile, 1);
    tile = std::min(tile, nq);
    return tile;
}

bool allowCpuFallbackForIvf() {
    const char* env = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    if (!env || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

void floatToHalf(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        __fp16 h = (__fp16)src[i];
        std::memcpy(&dst[i], &h, sizeof(uint16_t));
    }
}

inline faiss::idx_t decodeCpuLabelFromPair(
        const faiss::IndexIVFFlat* cpuIndex,
        int64_t pairLabel) {
    const uint64_t pair = static_cast<uint64_t>(pairLabel);
    const uint64_t listNo = faiss::lo_listno(pair);
    const uint64_t offset = faiss::lo_offset(pair);
    if (!cpuIndex || listNo >= (uint64_t)cpuIndex->nlist) {
        return -1;
    }
    const size_t sz = cpuIndex->invlists->list_size((size_t)listNo);
    if (offset >= sz) {
        return -1;
    }
    const faiss::idx_t* ids = cpuIndex->invlists->get_ids((size_t)listNo);
    return ids ? ids[offset] : -1;
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
        : MetalIndex(resources, dims, metric, metricArg, config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
    // Simple CPU quantizer: IndexFlatL2 or IndexFlatIP
    faiss::IndexFlat* quantizer = (metric == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dims)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dims);
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, (size_t)d, (size_t)nlist, metric);
    cpuIndex_->own_fields = true;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)d,
            nlist,
            metric,
            metricArg,
            indicesOptions_,
            interleavedLayout_);
}

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config,
        bool ownFields)
        : MetalIndex(resources, dims, metric, metricArg, config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
    FAISS_THROW_IF_NOT_MSG(
            coarseQuantizer != nullptr,
            "MetalIndexIVFFlat: coarseQuantizer must be non-null");
    cpuIndex_ = std::make_unique<faiss::IndexIVFFlat>(
            coarseQuantizer, (size_t)d, (size_t)nlist, metric);
    cpuIndex_->own_fields = ownFields;
    gpuIvf_ = std::make_unique<MetalIVFFlatImpl>(
            resources,
            (int)d,
            nlist,
            metric,
            metricArg,
            indicesOptions_,
            interleavedLayout_);
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
                  config),
          indicesOptions_(config.indicesOptions),
          interleavedLayout_(config.interleavedLayout) {
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
            cpuIndex->metric_arg,
            indicesOptions_,
            interleavedLayout_);
    copyFrom(cpuIndex);
}

MetalIndexIVFFlat::~MetalIndexIVFFlat() = default;

void MetalIndexIVFFlat::uploadCentroids_() const {
    if (!cpuIndex_ || !cpuIndex_->quantizer || !resources_) {
        return;
    }
    auto* q = cpuIndex_->quantizer;
    if (!q || q->ntotal == 0) {
        centroidBuf_ = nil;
        centroidNormsBuf_ = nil;
        return;
    }
    size_t nCentroids = (size_t)q->ntotal;
    std::vector<float> centroids(nCentroids * (size_t)d);
    q->reconstruct_n(0, (idx_t)nCentroids, centroids.data());

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
        const float* src = centroids.data();
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
                                      (int)nCentroids, d, centroidNormsBuf_, false)) {
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
    const idx_t autoReserveMinBatch = getIvfAutoReserveMinBatch();
    if (gpuIvf_ && autoReserveMinBatch > 0 && n >= autoReserveMinBatch) {
        // Pre-reserve before append to reduce relayout spikes on large batches.
        gpuIvf_->reserveMemory(oldNt + n);
    }
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

    idx_t oldNt = cpuIndex_->ntotal;
    const idx_t autoReserveMinBatch = getIvfAutoReserveMinBatch();
    if (gpuIvf_ && autoReserveMinBatch > 0 && n >= autoReserveMinBatch) {
        // Pre-reserve before append to reduce relayout spikes on large batches.
        gpuIvf_->reserveMemory(oldNt + n);
    }

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
    FAISS_THROW_IF_NOT_MSG(
            k <= kIVFFlatSupportedMaxK,
            "MetalIndexIVFFlat supports k <= 1024; larger k is not yet supported");

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

    auto cpuFallbackSearch = [&](idx_t qBase, idx_t qCount) {
        const float* xTile = x + (size_t)qBase * (size_t)d;
        float* distancesTile = distances + (size_t)qBase * (size_t)k;
        idx_t* labelsTile = labels + (size_t)qBase * (size_t)k;
        if (indicesOptions_ == faiss::gpu::INDICES_IVF) {
            std::vector<float> coarseDist((size_t)qCount * nprobe);
            std::vector<idx_t> coarseAssign((size_t)qCount * nprobe);
            cpuIndex_->quantizer->search(
                    qCount,
                    xTile,
                    (idx_t)nprobe,
                    coarseDist.data(),
                    coarseAssign.data());
            cpuIndex_->search_preassigned(
                    qCount,
                    xTile,
                    k,
                    coarseAssign.data(),
                    coarseDist.data(),
                    distancesTile,
                    labelsTile,
                    true,
                    dynamic_cast<const IVFSearchParameters*>(params),
                    nullptr);
        } else {
            cpuIndex_->search(qCount, xTile, k, distancesTile, labelsTile);
        }
    };
    const bool allowCpuFallback = allowCpuFallbackForIvf();
    auto fallbackOrThrow = [&](idx_t qBase, idx_t qCount, const char* reason) {
        if (!allowCpuFallback) {
            FAISS_THROW_FMT(
                    "MetalIndexIVFFlat::search requires GPU execution but hit fallback (%s) "
                    "at qBase=%lld qCount=%lld",
                    reason,
                    (long long)qBase,
                    (long long)qCount);
        }
        cpuFallbackSearch(qBase, qCount);
    };

    id<MTLDevice>      device = resources_->getDevice();
    id<MTLCommandQueue> queue = resources_->getCommandQueue();
    if (gpuIvf_) {
        gpuIvf_->ensureInterleavedLayoutUpToDate();
    }

    const bool hasFlatCodes = gpuIvf_ && gpuIvf_->codesBuffer();
    const bool hasInterleavedCodes =
            gpuIvf_ && gpuIvf_->interleavedCodesBuffer() &&
            gpuIvf_->interleavedCodesOffsetBuffer();
    const bool hasScanCodes = hasFlatCodes || hasInterleavedCodes;

    // Fall back to CPU if Metal is not available or GPU IVF storage not ready.
    if (!device || !queue || !gpuIvf_ || !hasScanCodes || !gpuIvf_->idsBuffer() ||
        !gpuIvf_->listOffsetGpuBuffer() || !gpuIvf_->listLengthGpuBuffer()) {
        fallbackOrThrow(0, n, "missing Metal device/queue or IVF buffers");
        return;
    }

    // ---- Upload centroid buffer if needed -----------------------------------
    if (!centroidBuf_) {
        uploadCentroids_();
    }

    int nlist = (int)cpuIndex_->nlist;

    const size_t tileRows = chooseIvfSearchTileRows((size_t)n, d, k, nprobe, nlist);
    const int avgListLen = (ntotal > 0 && nlist > 0) ? (int)(ntotal / nlist) : 256;

    for (idx_t qBase = 0; qBase < n; qBase += (idx_t)tileRows) {
        idx_t qCount = std::min<idx_t>((idx_t)tileRows, n - qBase);
        const float* xTile = x + (size_t)qBase * (size_t)d;

        size_t queriesBytes = (size_t)qCount * (size_t)d * sizeof(float);
        size_t outDistBytes = (size_t)qCount * (size_t)k * sizeof(float);
        size_t outIdxBytes = (size_t)qCount * (size_t)k * sizeof(int64_t);
        size_t perListBytes = (size_t)qCount * nprobe * (size_t)k * sizeof(float);
        size_t perListIdxB = (size_t)qCount * nprobe * (size_t)k * sizeof(int64_t);
        size_t coarseDistBytes = (size_t)qCount * nprobe * sizeof(float);
        size_t coarseIdxBytes = (size_t)qCount * nprobe * sizeof(int32_t);
        size_t distMatBytes = (size_t)qCount * (size_t)nlist * sizeof(float);

        ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queriesBytes);
        ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
        ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
        ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
        ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
        ensureSearchBuf_(coarseOutDistBuf_, coarseOutDistCap_, coarseDistBytes);
        ensureSearchBuf_(coarseOutIdxBuf_, coarseOutIdxCap_, coarseIdxBytes);
        ensureSearchBuf_(distMatrixBuf_, distMatrixCap_, distMatBytes);

        if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
            !searchPerListDistBuf_ || !searchPerListIdxBuf_ ||
            !coarseOutDistBuf_ || !coarseOutIdxBuf_ || !distMatrixBuf_) {
            fallbackOrThrow(qBase, qCount, "failed to allocate tiled search buffers");
            continue;
        }

        std::memcpy([searchQueriesBuf_ contents], xTile, queriesBytes);

        bool ok = false;
        if (centroidBuf_ && nprobe <= (size_t)getMetalDistanceMaxK()) {
            ok = runMetalIVFFlatFullSearch(
                    device, queue,
                    searchQueriesBuf_,
                    (int)qCount, d, (int)k, (int)nprobe, isL2,
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
            std::vector<float> coarseDistVec((size_t)qCount * nprobe);
            std::vector<idx_t> coarseAssignVec((size_t)qCount * nprobe);
            cpuIndex_->quantizer->search(
                    qCount,
                    xTile,
                    (idx_t)nprobe,
                    coarseDistVec.data(),
                    coarseAssignVec.data());
            size_t coarseBytes = (size_t)qCount * nprobe * sizeof(int32_t);
            ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);
            if (!searchCoarseBuf_) {
                fallbackOrThrow(qBase, qCount, "failed to allocate coarse assign buffer");
                continue;
            }
            auto* dst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
            for (size_t i = 0; i < (size_t)qCount * nprobe; ++i) {
                FAISS_THROW_IF_NOT_MSG(
                        coarseAssignVec[i] >=
                                        (idx_t)std::numeric_limits<int32_t>::min() &&
                                coarseAssignVec[i] <=
                                        (idx_t)std::numeric_limits<int32_t>::max(),
                        "MetalIndexIVFFlat: coarse assignment exceeds int32 range");
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
                    (int)qCount, d, (int)k, (int)nprobe, isL2,
                    searchOutDistBuf_, searchOutIdxBuf_,
                    searchPerListDistBuf_, searchPerListIdxBuf_,
                    gpuIvf_->interleavedCodesBuffer(),
                    gpuIvf_->interleavedCodesOffsetBuffer());
        }

        if (!ok) {
            fallbackOrThrow(qBase, qCount, "GPU IVF scan failed");
            continue;
        }

        const float* outDistPtr = reinterpret_cast<const float*>(
                [searchOutDistBuf_ contents]);
        const int64_t* outIdxPtr = reinterpret_cast<const int64_t*>(
                [searchOutIdxBuf_ contents]);

        for (idx_t qi = 0; qi < qCount; ++qi) {
            for (idx_t j = 0; j < k; ++j) {
                size_t localPos = (size_t)qi * (size_t)k + (size_t)j;
                size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                int64_t globalId = outIdxPtr[localPos];
                if (globalId < 0) {
                    labels[globalPos] = -1;
                } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                    labels[globalPos] =
                            decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                    labels[globalPos] = (idx_t)(int32_t)globalId;
                } else {
                    labels[globalPos] = (idx_t)globalId;
                }
                distances[globalPos] = outDistPtr[localPos];
            }
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
    FAISS_THROW_IF_NOT(cpuIndex_);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(assign);
    FAISS_THROW_IF_NOT_MSG(stats == nullptr, "IVF stats not supported");
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs,
            "MetalIndexIVFFlat::search_preassigned does not currently support store_pairs");
    if (params) {
        FAISS_THROW_IF_NOT_FMT(
                params->max_codes == 0,
                "Metal IVF index does not currently support "
                "SearchParametersIVF::max_codes (passed %zu, must be 0)",
                params->max_codes);
    }
    FAISS_THROW_IF_NOT_MSG(
            k <= kIVFFlatSupportedMaxK,
            "MetalIndexIVFFlat supports k <= 1024; larger k is not yet supported");

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
    if (gpuIvf_) {
        gpuIvf_->ensureInterleavedLayoutUpToDate();
    }

    const bool hasFlatCodes = gpuIvf_ && gpuIvf_->codesBuffer();
    const bool hasInterleavedCodes =
            gpuIvf_ && gpuIvf_->interleavedCodesBuffer() &&
            gpuIvf_->interleavedCodesOffsetBuffer();
    const bool hasScanCodes = hasFlatCodes || hasInterleavedCodes;

    auto cpuFallbackSearch = [&](idx_t qBase, idx_t qCount) {
        const float* xTile = x + (size_t)qBase * (size_t)d;
        const idx_t* assignTile = assign + (size_t)qBase * nprobe;
        const float* centroidTile =
                centroid_dis ? (centroid_dis + (size_t)qBase * nprobe) : nullptr;
        float* distTile = distances + (size_t)qBase * (size_t)k;
        idx_t* labelsTile = labels + (size_t)qBase * (size_t)k;
        cpuIndex_->search_preassigned(
                qCount,
                xTile,
                k,
                assignTile,
                centroidTile,
                distTile,
                labelsTile,
                indicesOptions_ == faiss::gpu::INDICES_IVF ? true : store_pairs,
                params,
                stats);
    };
    const bool allowCpuFallback = allowCpuFallbackForIvf();
    auto fallbackOrThrow = [&](idx_t qBase, idx_t qCount, const char* reason) {
        if (!allowCpuFallback) {
            FAISS_THROW_FMT(
                    "MetalIndexIVFFlat::search_preassigned requires GPU execution but hit "
                    "fallback (%s) at qBase=%lld qCount=%lld",
                    reason,
                    (long long)qBase,
                    (long long)qCount);
        }
        cpuFallbackSearch(qBase, qCount);
    };

    if (!device || !queue || !gpuIvf_ || !hasScanCodes || !gpuIvf_->idsBuffer() ||
        !gpuIvf_->listOffsetGpuBuffer() || !gpuIvf_->listLengthGpuBuffer()) {
        fallbackOrThrow(0, n, "missing Metal device/queue or IVF buffers");
        return;
    }

    const size_t tileRows = chooseIvfPreassignedTileRows((size_t)n, d, k, nprobe);
    for (idx_t qBase = 0; qBase < n; qBase += (idx_t)tileRows) {
        idx_t qCount = std::min<idx_t>((idx_t)tileRows, n - qBase);
        const float* xTile = x + (size_t)qBase * (size_t)d;
        const idx_t* assignTile = assign + (size_t)qBase * nprobe;

        size_t queriesBytes = (size_t)qCount * (size_t)d * sizeof(float);
        size_t outDistBytes = (size_t)qCount * (size_t)k * sizeof(float);
        size_t outIdxBytes = (size_t)qCount * (size_t)k * sizeof(int64_t);
        size_t perListBytes = (size_t)qCount * nprobe * (size_t)k * sizeof(float);
        size_t perListIdxB = (size_t)qCount * nprobe * (size_t)k * sizeof(int64_t);
        size_t coarseBytes = (size_t)qCount * nprobe * sizeof(int32_t);

        ensureSearchBuf_(searchQueriesBuf_, searchQueriesCap_, queriesBytes);
        ensureSearchBuf_(searchOutDistBuf_, searchOutDistCap_, outDistBytes);
        ensureSearchBuf_(searchOutIdxBuf_, searchOutIdxCap_, outIdxBytes);
        ensureSearchBuf_(searchPerListDistBuf_, searchPerListDistCap_, perListBytes);
        ensureSearchBuf_(searchPerListIdxBuf_, searchPerListIdxCap_, perListIdxB);
        ensureSearchBuf_(searchCoarseBuf_, searchCoarseCap_, coarseBytes);

        if (!searchQueriesBuf_ || !searchOutDistBuf_ || !searchOutIdxBuf_ ||
            !searchPerListDistBuf_ || !searchPerListIdxBuf_ || !searchCoarseBuf_) {
            fallbackOrThrow(qBase, qCount, "failed to allocate tiled preassigned buffers");
            continue;
        }

        std::memcpy([searchQueriesBuf_ contents], xTile, queriesBytes);

        auto* coarseDst = reinterpret_cast<int32_t*>([searchCoarseBuf_ contents]);
        for (size_t i = 0; i < (size_t)qCount * nprobe; ++i) {
            FAISS_THROW_IF_NOT_MSG(
                    assignTile[i] >= (idx_t)std::numeric_limits<int32_t>::min() &&
                            assignTile[i] <=
                                    (idx_t)std::numeric_limits<int32_t>::max(),
                    "MetalIndexIVFFlat: preassigned list id exceeds int32 range");
            coarseDst[i] = (int32_t)assignTile[i];
        }

        bool ok = runMetalIVFFlatScan(
                device, queue,
                searchQueriesBuf_,
                gpuIvf_->codesBuffer(),
                gpuIvf_->idsBuffer(),
                gpuIvf_->listOffsetGpuBuffer(),
                gpuIvf_->listLengthGpuBuffer(),
                searchCoarseBuf_,
                (int)qCount, d, (int)k, (int)nprobe, isL2,
                searchOutDistBuf_, searchOutIdxBuf_,
                searchPerListDistBuf_, searchPerListIdxBuf_,
                gpuIvf_->interleavedCodesBuffer(),
                gpuIvf_->interleavedCodesOffsetBuffer());

        if (!ok) {
            fallbackOrThrow(qBase, qCount, "GPU IVF scan failed");
            continue;
        }

        const float* outDistPtr = reinterpret_cast<const float*>(
                [searchOutDistBuf_ contents]);
        const int64_t* outIdxPtr = reinterpret_cast<const int64_t*>(
                [searchOutIdxBuf_ contents]);
        for (idx_t qi = 0; qi < qCount; ++qi) {
            for (idx_t j = 0; j < k; ++j) {
                size_t localPos = (size_t)qi * (size_t)k + (size_t)j;
                size_t globalPos = (size_t)(qBase + qi) * (size_t)k + (size_t)j;
                int64_t globalId = outIdxPtr[localPos];
                if (globalId < 0) {
                    labels[globalPos] = -1;
                } else if (indicesOptions_ == faiss::gpu::INDICES_CPU) {
                    labels[globalPos] =
                            decodeCpuLabelFromPair(cpuIndex_.get(), globalId);
                } else if (indicesOptions_ == faiss::gpu::INDICES_32_BIT) {
                    labels[globalPos] = (idx_t)(int32_t)globalId;
                } else {
                    labels[globalPos] = (idx_t)globalId;
                }
                distances[globalPos] = outDistPtr[localPos];
            }
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

bool MetalIndexIVFFlat::interleavedLayout() const {
    return interleavedLayout_;
}

faiss::gpu::IndicesOptions MetalIndexIVFFlat::indicesOptions() const {
    return indicesOptions_;
}

MetalIndexIVFFlat::AppendDebugStats MetalIndexIVFFlat::appendDebugStats() const {
    AppendDebugStats out{};
    if (!gpuIvf_) {
        return out;
    }
    const auto& s = gpuIvf_->appendDebugStats();
    out.relayoutEvents = s.relayoutEvents;
    out.movedLists = s.movedLists;
    out.movedVectors = s.movedVectors;
    out.reusedSegmentAllocs = s.reusedSegmentAllocs;
    out.tailSegmentAllocs = s.tailSegmentAllocs;
    out.reusedCapacityVecs = s.reusedCapacityVecs;
    out.tailCapacityVecs = s.tailCapacityVecs;
    out.tailShrinkEvents = s.tailShrinkEvents;
    out.tailShrunkVecs = s.tailShrunkVecs;
    return out;
}

void MetalIndexIVFFlat::resetAppendDebugStats() {
    if (gpuIvf_) {
        gpuIvf_->resetAppendDebugStats();
    }
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

    // Copy quantizer centroids (allow non-IndexFlat CPU coarse quantizers by
    // reconstructing centroid vectors).
    FAISS_THROW_IF_NOT_MSG(src->quantizer, "copyFrom: source quantizer is null");
    auto* ourQ = cpuIndex_->quantizer;
    FAISS_THROW_IF_NOT_MSG(ourQ, "copyFrom: internal quantizer is null");
    ourQ->reset();
    if (src->nlist > 0) {
        std::vector<float> coarse((size_t)src->nlist * d);
        src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
        if (!ourQ->is_trained) {
            ourQ->train(src->nlist, coarse.data());
        }
        ourQ->add(src->nlist, coarse.data());
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

    auto* srcQ = cpuIndex_->quantizer;
    auto* dstQ = dst->quantizer;
    FAISS_THROW_IF_NOT_MSG(srcQ, "copyTo: internal quantizer is null");
    FAISS_THROW_IF_NOT_MSG(dstQ, "copyTo: destination quantizer is null");

    dstQ->reset();
    if (srcQ->ntotal > 0) {
        std::vector<float> coarse((size_t)srcQ->ntotal * d);
        srcQ->reconstruct_n(0, srcQ->ntotal, coarse.data());
        if (!dstQ->is_trained) {
            dstQ->train(srcQ->ntotal, coarse.data());
        }
        dstQ->add(srcQ->ntotal, coarse.data());
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
