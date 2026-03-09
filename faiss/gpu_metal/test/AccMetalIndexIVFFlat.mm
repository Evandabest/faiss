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
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
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
