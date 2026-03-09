// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Correctness tests for MetalIndexBinaryFlat: compare GPU Hamming
 * search against CPU IndexBinaryFlat.
 */

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <gtest/gtest.h>
#import <cstdlib>
#import <memory>
#import <set>
#import <vector>

namespace {

void fillRandom(uint8_t* data, size_t n, unsigned seed) {
    std::srand(seed);
    for (size_t i = 0; i < n; ++i)
        data[i] = (uint8_t)(std::rand() & 0xFF);
}

float computeRecall(
        int nq,
        int k,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    int hits = 0, total = 0;
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

} // namespace

class AccMetalIndexBinaryFlat : public ::testing::Test {
protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        ASSERT_TRUE(resources_->isAvailable()) << "Metal not available";
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(AccMetalIndexBinaryFlat, BasicD256) {
    const int d = 256, nb = 5000, nq = 32, k = 10;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 42);
    fillRandom(xq.data(), xq.size(), 123);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, &cpuIdx);

    std::vector<int32_t> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx.search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.99f) << "D=256 recall = " << recall;

    for (int q = 0; q < nq; ++q) {
        EXPECT_EQ(cpuDist[q * k], metalDist[q * k])
                << "Top-1 distance mismatch for query " << q;
    }
}

TEST_F(AccMetalIndexBinaryFlat, BasicD128) {
    const int d = 128, nb = 3000, nq = 16, k = 5;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 7);
    fillRandom(xq.data(), xq.size(), 77);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, &cpuIdx);

    std::vector<int32_t> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx.search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.99f) << "D=128 recall = " << recall;
}

TEST_F(AccMetalIndexBinaryFlat, D64) {
    const int d = 64, nb = 2000, nq = 20, k = 10;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 99);
    fillRandom(xq.data(), xq.size(), 100);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, d);
    metalIdx.add(nb, xb.data());

    std::vector<int32_t> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx.search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.99f) << "D=64 recall = " << recall;
}

TEST_F(AccMetalIndexBinaryFlat, D1024) {
    const int d = 1024, nb = 2000, nq = 10, k = 5;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 55);
    fillRandom(xq.data(), xq.size(), 56);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, &cpuIdx);

    std::vector<int32_t> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);

    cpuIdx.search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.99f) << "D=1024 recall = " << recall;
}

TEST_F(AccMetalIndexBinaryFlat, Reconstruct) {
    const int d = 256, nb = 100;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs);
    fillRandom(xb.data(), xb.size(), 1234);

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, d);
    metalIdx.add(nb, xb.data());

    for (int i = 0; i < nb; i += 10) {
        std::vector<uint8_t> recons(cs);
        metalIdx.reconstruct(i, recons.data());
        for (int j = 0; j < cs; ++j) {
            EXPECT_EQ(recons[j], xb[i * cs + j])
                    << "Mismatch at vector " << i << " byte " << j;
        }
    }
}

TEST_F(AccMetalIndexBinaryFlat, CopyFromTo) {
    const int d = 256, nb = 1000, nq = 10, k = 5;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 5678);
    fillRandom(xq.data(), xq.size(), 9012);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat metalIdx(resources_, &cpuIdx);

    std::vector<int32_t> metalDist(nq * k);
    std::vector<faiss::idx_t> metalLab(nq * k);
    metalIdx.search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    faiss::IndexBinaryFlat cpuIdx2(d);
    metalIdx.copyTo(&cpuIdx2);

    std::vector<int32_t> cpu2Dist(nq * k);
    std::vector<faiss::idx_t> cpu2Lab(nq * k);
    cpuIdx2.search(nq, xq.data(), k, cpu2Dist.data(), cpu2Lab.data());

    float recall = computeRecall(nq, k, metalLab.data(), cpu2Lab.data());
    EXPECT_GE(recall, 0.99f) << "CopyTo round-trip recall = " << recall;
}

TEST_F(AccMetalIndexBinaryFlat, ClonerRoundTrip) {
    const int d = 256, nb = 1000, nq = 10, k = 5;
    const int cs = d / 8;
    std::vector<uint8_t> xb(nb * cs), xq(nq * cs);
    fillRandom(xb.data(), xb.size(), 3456);
    fillRandom(xq.data(), xq.size(), 7890);

    faiss::IndexBinaryFlat cpuIdx(d);
    cpuIdx.add(nb, xb.data());

    faiss::gpu_metal::StandardMetalResources stdRes;

    faiss::IndexBinary* metalRaw =
            faiss::gpu_metal::index_binary_cpu_to_metal_gpu(
                    &stdRes, 0, &cpuIdx);
    ASSERT_NE(metalRaw, nullptr);

    std::vector<int32_t> metalDist(nq * k);
    std::vector<faiss::idx_t> metalLab(nq * k);
    metalRaw->search(nq, xq.data(), k, metalDist.data(), metalLab.data());

    faiss::IndexBinary* cpuBack =
            faiss::gpu_metal::index_binary_metal_gpu_to_cpu(metalRaw);
    ASSERT_NE(cpuBack, nullptr);

    std::vector<int32_t> cpuDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k);
    cpuBack->search(nq, xq.data(), k, cpuDist.data(), cpuLab.data());

    float recall = computeRecall(nq, k, metalLab.data(), cpuLab.data());
    EXPECT_GE(recall, 0.99f) << "Cloner round-trip recall = " << recall;

    delete metalRaw;
    delete cpuBack;
}
