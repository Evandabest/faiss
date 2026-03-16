// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Correctness tests for MetalIndexIVFPQ: compare GPU results against CPU
 * IndexIVFPQ for L2 and inner-product metrics.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFPQ.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <set>
#import <vector>

namespace {

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
            if (lab >= 0) refSet.insert(lab);
        }
        for (int i = 0; i < k; ++i) {
            faiss::idx_t lab = testLab[q * k + i];
            if (lab >= 0 && refSet.count(lab)) hits++;
        }
        total += (int)refSet.size();
    }
    return total > 0 ? (float)hits / (float)total : 1.0f;
}

std::unique_ptr<faiss::IndexIVFPQ> makeCpuIVFPQ(
        int dim,
        int nlist,
        int M,
        int nbits,
        faiss::MetricType metric,
        int nb,
        const float* trainData) {
    faiss::IndexFlat* quantizer = (metric == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dim)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dim);
    auto idx = std::make_unique<faiss::IndexIVFPQ>(
            quantizer, (size_t)dim, (size_t)nlist, (size_t)M, (size_t)nbits);
    idx->own_fields = true;
    idx->train(nb, trainData);
    return idx;
}

} // namespace

class AccMetalIndexIVFPQ : public ::testing::Test {
protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        ASSERT_TRUE(resources_->isAvailable()) << "Metal not available";
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(AccMetalIndexIVFPQ, L2_M8) {
    const int d = 64, nlist = 16, nb = 5000, nq = 32, k = 10, M = 8;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 42);
    faiss::float_rand(xq.data(), xq.size(), 123);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.85f) << "L2 M=8 recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, IP_M8) {
    const int d = 64, nlist = 16, nb = 5000, nq = 32, k = 10, M = 8;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 42);
    faiss::float_rand(xq.data(), xq.size(), 123);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_INNER_PRODUCT, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.85f) << "IP M=8 recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, L2_M4_D128) {
    const int d = 128, nlist = 32, nb = 10000, nq = 20, k = 5, M = 4;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 7);
    faiss::float_rand(xq.data(), xq.size(), 77);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.85f) << "L2 M=4 D=128 recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, L2_M16_D128) {
    const int d = 128, nlist = 32, nb = 10000, nq = 20, k = 10, M = 16;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 13);
    faiss::float_rand(xq.data(), xq.size(), 37);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.85f) << "L2 M=16 D=128 recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, L2_M16_D128_Fp16Lut) {
    const int d = 128, nlist = 32, nb = 10000, nq = 20, k = 10, M = 16;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 113);
    faiss::float_rand(xq.data(), xq.size(), 137);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 8;

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.80f) << "L2 M=16 D=128 fp16 LUT recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, CopyFromTo) {
    const int d = 64, nlist = 16, nb = 3000, nq = 10, k = 5, M = 8;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 99);
    faiss::float_rand(xq.data(), xq.size(), 100);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, cpuIdx.get(), config);

    // Search on Metal.
    std::vector<float> metalDist(nq * k);
    std::vector<faiss::idx_t> metalLab(nq * k);
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    // Copy back to CPU.
    faiss::IndexFlat* q2 = new faiss::IndexFlatL2(d);
    faiss::IndexIVFPQ cpuIdx2(q2, d, nlist, M, 8);
    cpuIdx2.own_fields = true;
    metalIdx.copyTo(&cpuIdx2);
    cpuIdx2.nprobe = 4;

    std::vector<float> cpu2Dist(nq * k);
    std::vector<faiss::idx_t> cpu2Lab(nq * k);
    cpuIdx2.search(nq, xq.data(), k, cpu2Dist.data(), cpu2Lab.data());

    float recall = computeRecall(nq, k, metalLab.data(), cpu2Lab.data());
    EXPECT_GE(recall, 0.95f) << "CopyTo round-trip recall = " << recall;
}

TEST_F(AccMetalIndexIVFPQ, ClonerRoundTrip) {
    const int d = 64, nlist = 16, nb = 3000, nq = 10, k = 5, M = 8;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 55);
    faiss::float_rand(xq.data(), xq.size(), 56);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;

    faiss::gpu_metal::StandardMetalResources stdRes;

    faiss::Index* metalRaw =
            faiss::gpu_metal::index_cpu_to_metal_gpu(&stdRes, 0, cpuIdx.get());
    ASSERT_NE(metalRaw, nullptr);

    std::vector<float> metalDist(nq * k);
    std::vector<faiss::idx_t> metalLab(nq * k);
    metalRaw->search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    faiss::Index* cpuBack =
            faiss::gpu_metal::index_metal_gpu_to_cpu(metalRaw);
    ASSERT_NE(cpuBack, nullptr);

    std::vector<float> cpuDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k);
    cpuBack->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());

    float recall = computeRecall(nq, k, metalLab.data(), cpuLab.data());
    EXPECT_GE(recall, 0.95f) << "Cloner round-trip recall = " << recall;

    delete metalRaw;
    delete cpuBack;
}

TEST_F(AccMetalIndexIVFPQ, ClonerOptionsUsePrecomputedIVFPQ) {
    const int d = 64, nlist = 16, nb = 4000, M = 8;
    std::vector<float> xb(nb * d);
    faiss::float_rand(xb.data(), xb.size(), 777);

    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;
    cpuIdx->use_precomputed_table = 0;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.usePrecomputed = true;

    std::unique_ptr<faiss::Index> metalRaw(
            faiss::gpu_metal::index_cpu_to_metal_gpu(
                    &stdRes, 0, cpuIdx.get(), &opts));
    ASSERT_NE(metalRaw.get(), nullptr);
    auto* metal = dynamic_cast<faiss::gpu_metal::MetalIndexIVFPQ*>(metalRaw.get());
    ASSERT_NE(metal, nullptr);
    EXPECT_TRUE(metal->getUsePrecomputedTables());

    std::unique_ptr<faiss::Index> cpuBack(
            faiss::gpu_metal::index_metal_gpu_to_cpu(metalRaw.get()));
    ASSERT_NE(cpuBack.get(), nullptr);
    auto* ivfpqBack = dynamic_cast<faiss::IndexIVFPQ*>(cpuBack.get());
    ASSERT_NE(ivfpqBack, nullptr);
    EXPECT_EQ(ivfpqBack->use_precomputed_table, 1);
}

TEST_F(AccMetalIndexIVFPQ, TrainAndAdd) {
    const int d = 32, nlist = 8, nb = 2000, nq = 16, k = 10, M = 4;
    std::vector<float> xb(nb * d), xq(nq * d);
    faiss::float_rand(xb.data(), xb.size(), 1234);
    faiss::float_rand(xq.data(), xq.size(), 5678);

    faiss::gpu_metal::MetalIndexConfig config;
    faiss::gpu_metal::MetalIndexIVFPQ metalIdx(
            resources_, d, nlist, M, 8, faiss::METRIC_L2, 0.0f, config);

    metalIdx.train(nb, xb.data());
    ASSERT_TRUE(metalIdx.is_trained);
    metalIdx.add(nb, xb.data());
    EXPECT_EQ(metalIdx.ntotal, nb);

    // Compare against CPU.
    auto cpuIdx = makeCpuIVFPQ(d, nlist, M, 8, faiss::METRIC_L2, nb, xb.data());
    cpuIdx->add(nb, xb.data());
    cpuIdx->nprobe = 4;

    // Set same nprobe on metal via the internal search params.
    faiss::IVFSearchParameters sp;
    sp.nprobe = 4;

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data(), &sp);

    for (int qi = 0; qi < nq; ++qi) {
        bool anyResult = false;
        for (int j = 0; j < k; ++j) {
            if (metalLab[qi * k + j] >= 0) anyResult = true;
        }
        EXPECT_TRUE(anyResult) << "Query " << qi << " returned no results";
    }
}
