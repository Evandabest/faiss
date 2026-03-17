// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Correctness tests for MetalIndexIVFFlat: compare GPU results against CPU
 * IndexIVFFlat for L2 and inner-product metrics.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <limits>
#import <memory>
#import <set>
#import <vector>

namespace {

/// Compute recall: fraction of CPU top-k labels present in Metal top-k.
float computeRecall(
        int nq,
        int k,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    int hits = 0;
    int total = 0;
    for (int q = 0; q < nq; ++q) {
        std::set<faiss::idx_t> refSet;
        for (int i = 0; i < k; ++i) {
            faiss::idx_t lab = refLab[q * k + i];
            if (lab >= 0) {
                refSet.insert(lab);
            }
        }
        for (int i = 0; i < k; ++i) {
            faiss::idx_t lab = testLab[q * k + i];
            if (lab >= 0 && refSet.count(lab)) {
                hits++;
            }
        }
        total += (int)refSet.size();
    }
    return total > 0 ? (float)hits / (float)total : 1.0f;
}

/// Assert recall >= threshold.
void expectRecall(
        int nq,
        int k,
        float threshold,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    float recall = computeRecall(nq, k, refLab, testLab);
    EXPECT_GE(recall, threshold)
            << "Recall " << recall << " below threshold " << threshold;
}

/// Build and train a CPU IndexIVFFlat (owns its quantizer).
std::unique_ptr<faiss::IndexIVFFlat> makeCpuIVFFlat(
        int dim,
        int nlist,
        faiss::MetricType metric,
        int nb,
        const float* trainData) {
    faiss::IndexFlat* quantizer = (metric == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dim)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dim);
    auto idx = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, (size_t)dim, (size_t)nlist, metric);
    idx->own_fields = true;
    idx->train(nb, trainData);
    return idx;
}

std::unique_ptr<faiss::IndexIVFScalarQuantizer> makeCpuIVFSQFromIVFFlat(
        const faiss::IndexIVFFlat* src,
        faiss::ScalarQuantizer::QuantizerType sqType) {
    faiss::IndexFlat* quantizer = (src->metric_type == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP((int)src->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2((int)src->d);
    auto dst = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            quantizer,
            src->d,
            src->nlist,
            sqType,
            src->metric_type,
            false);
    dst->own_fields = true;
    dst->metric_arg = src->metric_arg;

    std::vector<float> coarse((size_t)src->nlist * src->d);
    src->quantizer->reconstruct_n(0, src->nlist, coarse.data());
    dst->quantizer->train(src->nlist, coarse.data());
    dst->quantizer->add(src->nlist, coarse.data());

    size_t totalN = 0;
    for (size_t l = 0; l < (size_t)src->nlist; ++l) {
        totalN += src->invlists->list_size(l);
    }
    std::vector<float> allVecs(totalN * (size_t)src->d);
    std::vector<faiss::idx_t> allIds(totalN);
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

    if (totalN > 0) {
        dst->train((faiss::idx_t)totalN, allVecs.data());
        dst->add_with_ids((faiss::idx_t)totalN, allVecs.data(), allIds.data());
    }
    dst->nprobe = src->nprobe;
    return dst;
}

} // namespace

class AccMetalIndexIVFFlat : public ::testing::Test {
protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

// ---------- L2 ----------

TEST_F(AccMetalIndexIVFFlat, L2_Basic) {
    const int dim = 64, nb = 5000, nq = 50, nlist = 32, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 1002);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    // Use same nprobe via search params
    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, L2_HighProbe) {
    const int dim = 32, nb = 3000, nq = 30, nlist = 16, k = 20;
    const size_t nprobe = 16; // exhaustive probe

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 2001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 2002);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    // nprobe == nlist → full scan, should match very closely
    expectRecall(nq, k, 0.95f, refL.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, L2_SmallK) {
    const int dim = 64, nb = 5000, nq = 40, nlist = 32, k = 1;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 3001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 3002);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, L2_LargeK) {
    const int dim = 64, nb = 5000, nq = 20, nlist = 32, k = 50;
    const size_t nprobe = 12;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 4001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 4002);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
}

// ---------- Inner product ----------

TEST_F(AccMetalIndexIVFFlat, IP_Basic) {
    const int dim = 64, nb = 5000, nq = 50, nlist = 32, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 5001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5002);

    auto cpuIdx = makeCpuIVFFlat(
            dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist,
            faiss::METRIC_INNER_PRODUCT);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, IP_HighProbe) {
    const int dim = 32, nb = 3000, nq = 30, nlist = 16, k = 20;
    const size_t nprobe = 16;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 6001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 6002);

    auto cpuIdx = makeCpuIVFFlat(
            dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist,
            faiss::METRIC_INNER_PRODUCT);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.95f, refL.data(), testL.data());
}

// ---------- Edge cases ----------

TEST_F(AccMetalIndexIVFFlat, EmptyIndex) {
    const int dim = 32, nq = 5, nlist = 8, k = 3;

    std::vector<float> trainVecs((size_t)500 * dim);
    faiss::float_rand(trainVecs.data(), trainVecs.size(), 7001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 7002);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(500, trainVecs.data());

    std::vector<float> dists((size_t)nq * k);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -2);
    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());

    for (int i = 0; i < nq * k; ++i) {
        EXPECT_EQ(labels[i], -1) << "empty index should return -1 labels";
    }
}

TEST_F(AccMetalIndexIVFFlat, RejectsKAbove1024InSearch) {
    const int dim = 32, nlist = 16, nq = 2;
    const faiss::idx_t k = 1025;

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);

    std::vector<float> queries((size_t)nq * dim, 0.0f);
    std::vector<float> distances((size_t)nq * (size_t)k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * (size_t)k, -1);

    EXPECT_ANY_THROW(
            metalIdx.search(nq, queries.data(), k, distances.data(), labels.data()));
}

TEST_F(AccMetalIndexIVFFlat, RejectsKAbove1024InSearchPreassigned) {
    const int dim = 32, nlist = 16, nq = 2;
    const faiss::idx_t k = 1025;
    const int nprobe = 4;

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);

    std::vector<float> queries((size_t)nq * dim, 0.0f);
    std::vector<float> distances((size_t)nq * (size_t)k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * (size_t)k, -1);
    std::vector<faiss::idx_t> assign((size_t)nq * (size_t)nprobe, 0);
    std::vector<float> centroidDistances((size_t)nq * (size_t)nprobe, 0.0f);

    faiss::IVFSearchParameters params;
    params.nprobe = nprobe;

    EXPECT_ANY_THROW(metalIdx.search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDistances.data(),
            distances.data(),
            labels.data(),
            false,
            &params,
            nullptr));
}

TEST_F(AccMetalIndexIVFFlat, ResetThenSearch) {
    const int dim = 32, nb = 1000, nq = 5, nlist = 8, k = 3;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8002);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());
    EXPECT_EQ(metalIdx.ntotal, nb);

    metalIdx.reset();
    EXPECT_EQ(metalIdx.ntotal, 0);

    std::vector<float> dists((size_t)nq * k);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -2);
    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());

    for (int i = 0; i < nq * k; ++i) {
        EXPECT_EQ(labels[i], -1) << "after reset, labels should be -1";
    }
}

TEST_F(AccMetalIndexIVFFlat, AddWithIds) {
    const int dim = 32, nb = 2000, nq = 20, nlist = 16, k = 5;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 9002);

    std::vector<faiss::idx_t> ids(nb);
    for (int i = 0; i < nb; ++i) {
        ids[i] = 10000 + (faiss::idx_t)i;
    }

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add_with_ids(nb, vecs.data(), ids.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> dists((size_t)nq * k);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data(), &ivfParams);

    for (int i = 0; i < nq * k; ++i) {
        if (labels[i] >= 0) {
            EXPECT_GE(labels[i], 10000) << "returned ids should use custom ids";
            EXPECT_LT(labels[i], 10000 + nb);
        }
    }
}

// ---------- Cloning ----------

TEST_F(AccMetalIndexIVFFlat, CpuToMetalGpu) {
    const int dim = 64, nb = 5000, nq = 30, nlist = 32, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 10001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 10002);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalRaw =
            faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, cpuIdx.get());
    ASSERT_NE(metalRaw, nullptr);
    EXPECT_EQ(metalRaw->ntotal, nb);

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, MetalGpuToCpu) {
    const int dim = 64, nb = 5000, nq = 30, nlist = 32, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 11001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 11002);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::Index* cpuRaw = faiss::gpu_metal::index_metal_gpu_to_cpu(&metalIdx);
    ASSERT_NE(cpuRaw, nullptr);
    EXPECT_EQ(cpuRaw->ntotal, nb);

    auto* cpuIVF = dynamic_cast<faiss::IndexIVFFlat*>(cpuRaw);
    ASSERT_NE(cpuIVF, nullptr);
    cpuIVF->nprobe = nprobe;

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    metalIdx.search(nq, queries.data(), k, refD.data(), refL.data(), &ivfParams);
    cpuIVF->search(nq, queries.data(), k, testD.data(), testL.data());

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());

    delete cpuRaw;
}

TEST_F(AccMetalIndexIVFFlat, RoundTripClone) {
    const int dim = 32, nb = 3000, nq = 20, nlist = 16, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 12001);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 12002);

    // CPU → Metal → CPU round trip
    auto origCpu = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    origCpu->nprobe = nprobe;
    origCpu->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalRaw =
            faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, origCpu.get());
    ASSERT_NE(metalRaw, nullptr);

    faiss::Index* backCpu =
            faiss::gpu_metal::index_metal_gpu_to_cpu(metalRaw);
    ASSERT_NE(backCpu, nullptr);
    EXPECT_EQ(backCpu->ntotal, nb);

    auto* backIVF = dynamic_cast<faiss::IndexIVFFlat*>(backCpu);
    ASSERT_NE(backIVF, nullptr);
    backIVF->nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    origCpu->search(nq, queries.data(), k, refD.data(), refL.data());
    backIVF->search(nq, queries.data(), k, testD.data(), testL.data());

    // Round-trip should be exact: same CPU index data
    for (int i = 0; i < nq * k; ++i) {
        EXPECT_EQ(refL[i], testL[i]) << "i=" << i;
        EXPECT_FLOAT_EQ(refD[i], testD[i]) << "i=" << i;
    }

    delete backCpu;
    delete metalRaw;
}

// ============================================================
//  Float16 coarse quantizer tests
// ============================================================

TEST_F(AccMetalIndexIVFFlat, FP16CoarseL2) {
    const int dim = 64;
    const int nlist = 32;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    std::vector<float> cpuD((size_t)nq * k);
    std::vector<faiss::idx_t> cpuL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16CoarseQuantizer = true;
    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> gpuD((size_t)nq * k);
    std::vector<faiss::idx_t> gpuL((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());

    expectRecall(nq, k, 0.85f, cpuL.data(), gpuL.data());
}

TEST_F(AccMetalIndexIVFFlat, FP16CoarseIP) {
    const int dim = 64;
    const int nlist = 32;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    std::vector<float> cpuD((size_t)nq * k);
    std::vector<faiss::idx_t> cpuL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16CoarseQuantizer = true;
    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> gpuD((size_t)nq * k);
    std::vector<faiss::idx_t> gpuL((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());

    expectRecall(nq, k, 0.85f, cpuL.data(), gpuL.data());
}

TEST_F(AccMetalIndexIVFFlat, FP16CoarseD128) {
    const int dim = 128;
    const int nlist = 64;
    const int nb = 10000;
    const int nq = 100;
    const int k = 20;
    const int nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    std::vector<float> cpuD((size_t)nq * k);
    std::vector<faiss::idx_t> cpuL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16CoarseQuantizer = true;
    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> gpuD((size_t)nq * k);
    std::vector<faiss::idx_t> gpuL((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());

    expectRecall(nq, k, 0.85f, cpuL.data(), gpuL.data());
}

// ---- D3 API tests: getListIndices / getListVectorData / updateQuantizer ----

TEST_F(AccMetalIndexIVFFlat, GetListIndices) {
    const int dim = 32, nlist = 8, nb = 500;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(resources_, cpuIdx.get());

    for (faiss::idx_t l = 0; l < nlist; ++l) {
        auto metalIds = metalIdx.getListIndices(l);
        size_t cpuLen = cpuIdx->invlists->list_size(l);
        EXPECT_EQ(metalIds.size(), cpuLen) << "List " << l;
        if (cpuLen > 0) {
            const faiss::idx_t* cpuIds = cpuIdx->invlists->get_ids(l);
            for (size_t i = 0; i < cpuLen; ++i)
                EXPECT_EQ(metalIds[i], cpuIds[i]);
        }
    }
}

TEST_F(AccMetalIndexIVFFlat, GetListVectorData) {
    const int dim = 16, nlist = 4, nb = 200;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 77);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(resources_, cpuIdx.get());

    for (faiss::idx_t l = 0; l < nlist; ++l) {
        auto metalVecs = metalIdx.getListVectorData(l);
        size_t cpuLen = cpuIdx->invlists->list_size(l);
        EXPECT_EQ(metalVecs.size(), cpuLen * (size_t)dim);
    }
}

TEST_F(AccMetalIndexIVFFlat, UpdateQuantizer) {
    const int dim = 32, nlist = 8, nb = 500, nq = 10, k = 5;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 123);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(resources_, cpuIdx.get());
    metalIdx.updateQuantizer();

    std::vector<float> dist(nq * k);
    std::vector<faiss::idx_t> lab(nq * k);
    metalIdx.search(nq, queries.data(), k, dist.data(), lab.data());

    bool anyResult = false;
    for (int i = 0; i < nq * k; ++i) {
        if (lab[i] >= 0) { anyResult = true; break; }
    }
    EXPECT_TRUE(anyResult);
}

TEST_F(AccMetalIndexIVFFlat, ReclaimMemory) {
    const int dim = 32, nlist = 4, nb = 100;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(resources_, cpuIdx.get());
    metalIdx.reclaimMemory();
    EXPECT_EQ(metalIdx.ntotal, nb);
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsFloat16Coarse) {
    const int dim = 64, nlist = 8, nb = 1000, nq = 10, k = 5;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 101);
    faiss::float_rand(queries.data(), queries.size(), 102);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.useFloat16CoarseQuantizer = true;
    opts.verbose = true;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    EXPECT_TRUE(metalRaw->verbose);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());
    metalRaw->search(nq, queries.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.70f) << "fp16 coarse recall = " << recall;

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsReserveVecs) {
    const int dim = 32, nlist = 4, nb = 200;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 103);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.reserveVecs = 10000;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    EXPECT_EQ(metalRaw->ntotal, nb);

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsInterleavedLayoutFalse) {
    const int dim = 64, nlist = 16, nb = 3000, nq = 20, k = 10;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 111);
    faiss::float_rand(queries.data(), queries.size(), 112);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = false;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIVF = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIVF, nullptr);
    EXPECT_FALSE(metalIVF->interleavedLayout());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.85f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsInterleavedLayoutTrue) {
    const int dim = 64, nlist = 16, nb = 3000, nq = 20, k = 10;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 311);
    faiss::float_rand(queries.data(), queries.size(), 312);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = true;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIVF = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIVF, nullptr);
    EXPECT_TRUE(metalIVF->interleavedLayout());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.85f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, InterleavedNonMultipleOf4DimensionMatchesCpu) {
    const int dim = 66, nlist = 16, nb = 3000, nq = 20, k = 10;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 411);
    faiss::float_rand(queries.data(), queries.size(), 412);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = true;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIVF = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIVF, nullptr);
    EXPECT_TRUE(metalIVF->interleavedLayout());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.99f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, DimensionAbove512MatchesCpu) {
    const int dim = 640, nlist = 16, nb = 2000, nq = 12, k = 8;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 421);
    faiss::float_rand(queries.data(), queries.size(), 422);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 6;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.99f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, KAtUpperBound1024MatchesCpu) {
    const int dim = 16, nlist = 1, nb = 512, nq = 4, k = 1024;

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 431);
    faiss::float_rand(queries.data(), queries.size(), 432);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 1;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());

    expectRecall(nq, k, 1.0f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, HighNprobeTimesKMatchesCpu) {
    const int dim = 32, nlist = 64, nb = 8000, nq = 24, k = 32;
    const size_t nprobe = 64; // nprobe * k = 2048 > 1024 exactness envelope

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 441);
    faiss::float_rand(queries.data(), queries.size(), 442);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());

    expectRecall(nq, k, 1.0f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, SkewedSingleListMatchesCpu) {
    const int dim = 24, nlist = 1, nb = 3000, nq = 20, k = 20;
    const size_t nprobe = 1; // single large list; list len > 1024

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 451);
    faiss::float_rand(queries.data(), queries.size(), 452);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k), testL((size_t)nq * k);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());

    expectRecall(nq, k, 1.0f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsIndicesIVF) {
    const int dim = 32, nlist = 8, nb = 2000, nq = 16, k = 8;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 211);
    faiss::float_rand(queries.data(), queries.size(), 212);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 6;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.indicesOptions = faiss::gpu::INDICES_IVF;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIVF = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIVF, nullptr);
    EXPECT_EQ(metalIVF->indicesOptions(), faiss::gpu::INDICES_IVF);

    std::vector<float> d((size_t)nq * k);
    std::vector<faiss::idx_t> l((size_t)nq * k, -1);
    metalRaw->search(nq, queries.data(), k, d.data(), l.data());

    for (faiss::idx_t lab : l) {
        if (lab < 0) {
            continue;
        }
        uint64_t pair = (uint64_t)lab;
        uint64_t listNo = faiss::lo_listno(pair);
        uint64_t offset = faiss::lo_offset(pair);
        EXPECT_LT(listNo, (uint64_t)nlist);
        if (listNo < (uint64_t)nlist) {
            EXPECT_LT(offset, cpuIdx->invlists->list_size((size_t)listNo));
        }
    }

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsIVFFlatToIVFSQ8) {
    const int dim = 48, nlist = 16, nb = 3000, nq = 20, k = 10;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1011);
    faiss::float_rand(queries.data(), queries.size(), 1012);

    auto cpuIVFFlat = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIVFFlat->add(nb, vecs.data());
    cpuIVFFlat->nprobe = 8;
    auto cpuIVFSQ = makeCpuIVFSQFromIVFFlat(
            cpuIVFFlat.get(), faiss::ScalarQuantizer::QT_8bit);

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.useIVFScalarQuantizer = true;
    opts.ivfSQType = faiss::ScalarQuantizer::QT_8bit;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIVFFlat.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalSQ = dynamic_cast<faiss::gpu_metal::MetalIndexIVFScalarQuantizer*>(metalRaw);
    ASSERT_NE(metalSQ, nullptr);
    EXPECT_EQ(metalSQ->sqQuantizerType(), faiss::ScalarQuantizer::QT_8bit);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIVFSQ->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.70f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ClonerOptionsIVFFlatToIVFSQFP16) {
    const int dim = 48, nlist = 16, nb = 3000, nq = 20, k = 10;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1111);
    faiss::float_rand(queries.data(), queries.size(), 1112);

    auto cpuIVFFlat = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIVFFlat->add(nb, vecs.data());
    cpuIVFFlat->nprobe = 8;
    auto cpuIVFSQ = makeCpuIVFSQFromIVFFlat(
            cpuIVFFlat.get(), faiss::ScalarQuantizer::QT_fp16);

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.useIVFScalarQuantizer = true;
    opts.ivfSQType = faiss::ScalarQuantizer::QT_fp16;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIVFFlat.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalSQ = dynamic_cast<faiss::gpu_metal::MetalIndexIVFScalarQuantizer*>(metalRaw);
    ASSERT_NE(metalSQ, nullptr);
    EXPECT_EQ(metalSQ->sqQuantizerType(), faiss::ScalarQuantizer::QT_fp16);

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIVFSQ->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.70f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, Indices32BitRejectsOutOfRangeIds) {
    const int dim = 16, nlist = 4, nb = 200;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 313);

    faiss::gpu_metal::MetalIndexConfig cfg;
    cfg.indicesOptions = faiss::gpu::INDICES_32_BIT;
    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, nlist, faiss::METRIC_L2, 0.0f, cfg);
    metalIdx.train(nb, vecs.data());

    std::vector<faiss::idx_t> ids(nb);
    for (int i = 0; i < nb; ++i) {
        ids[i] = (faiss::idx_t)std::numeric_limits<int32_t>::max() + 100 + i;
    }

    EXPECT_ANY_THROW(metalIdx.add_with_ids(nb, vecs.data(), ids.data()));
}

TEST_F(AccMetalIndexIVFFlat, UserProvidedCoarseQuantizer) {
    const int dim = 24, nlist = 8, nb = 1800, nq = 24, k = 8;
    const size_t nprobe = 4;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9011);
    faiss::float_rand(queries.data(), queries.size(), 9012);

    auto* cpuQ = new faiss::IndexScalarQuantizer(
            dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    auto cpuIdx = std::make_unique<faiss::IndexIVFFlat>(
            cpuQ, (size_t)dim, (size_t)nlist, faiss::METRIC_L2);
    cpuIdx->own_fields = true;
    cpuIdx->train(nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    auto* metalQ = new faiss::IndexScalarQuantizer(
            dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_,
            metalQ,
            dim,
            nlist,
            faiss::METRIC_L2,
            0.0f,
            faiss::gpu_metal::MetalIndexConfig(),
            true);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.70f, refL.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ClonerRejectsNonFlatCoarseQuantizerByDefault) {
    const int dim = 24, nlist = 8, nb = 1200;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 414);

    auto* sqQ = new faiss::IndexScalarQuantizer(
            dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    auto cpuIdx = std::make_unique<faiss::IndexIVFFlat>(
            sqQ, (size_t)dim, (size_t)nlist, faiss::METRIC_L2);
    cpuIdx->own_fields = true;
    cpuIdx->train(nb, vecs.data());
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    EXPECT_ANY_THROW(faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get()));
}

TEST_F(AccMetalIndexIVFFlat, ClonerAllowsNonFlatCoarseQuantizerWhenEnabled) {
    const int dim = 24, nlist = 8, nb = 1200, nq = 12, k = 6;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 515);
    faiss::float_rand(queries.data(), queries.size(), 516);

    auto* sqQ = new faiss::IndexScalarQuantizer(
            dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    auto cpuIdx = std::make_unique<faiss::IndexIVFFlat>(
            sqQ, (size_t)dim, (size_t)nlist, faiss::METRIC_L2);
    cpuIdx->own_fields = true;
    cpuIdx->train(nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = 4;

    std::vector<float> cpuD(nq * k);
    std::vector<faiss::idx_t> cpuL(nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.allowCpuCoarseQuantizer = true;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> testD(nq * k);
    std::vector<faiss::idx_t> testL(nq * k, -1);
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data());
    expectRecall(nq, k, 0.70f, cpuL.data(), testL.data());

    delete metalRaw;
}
