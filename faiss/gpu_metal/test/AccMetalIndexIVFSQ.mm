// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Correctness tests for MetalIndexIVFScalarQuantizer: compare GPU results
 * against CPU IndexIVFScalarQuantizer for QT_8bit and QT_fp16.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalIndexIVFScalarQuantizer.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>
#import <cmath>
#import <memory>
#import <random>
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

std::unique_ptr<faiss::IndexIVFScalarQuantizer> makeCpuIVFSQ(
        int dim,
        int nlist,
        faiss::ScalarQuantizer::QuantizerType sqType,
        faiss::MetricType metric,
        int nb,
        const float* trainData) {
    faiss::IndexFlat* quantizer = (metric == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(dim)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(dim);
    auto idx = std::make_unique<faiss::IndexIVFScalarQuantizer>(
            quantizer, (size_t)dim, (size_t)nlist, sqType, metric,
            /*by_residual=*/false);
    idx->own_fields = true;
    idx->train(nb, trainData);
    return idx;
}

struct IVFSQTestParam {
    int dim;
    int nlist;
    int nb;
    int nq;
    int k;
    faiss::ScalarQuantizer::QuantizerType sqType;
    faiss::MetricType metric;
    float recallThreshold;
};

class AccMetalIndexIVFSQ : public ::testing::TestWithParam<IVFSQTestParam> {};

TEST_P(AccMetalIndexIVFSQ, RecallVsCpu) {
    const auto& p = GetParam();

    auto res = std::make_shared<faiss::gpu_metal::MetalResources>();
    if (!res->isAvailable()) {
        GTEST_SKIP() << "Metal not available";
    }

    std::vector<float> data(p.nb * p.dim);
    std::vector<float> queries(p.nq * p.dim);
    if (p.sqType == faiss::ScalarQuantizer::QT_8bit_direct) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> u8(0, 255);
        for (float& v : data) {
            v = (float)u8(rng);
        }
        std::mt19937 rngQ(1337);
        for (float& v : queries) {
            v = (float)u8(rngQ);
        }
    } else {
        faiss::float_rand(data.data(), data.size(), 42);
        faiss::float_rand(queries.data(), queries.size(), 1337);
    }

    auto cpuIdx = makeCpuIVFSQ(
            p.dim, p.nlist, p.sqType, p.metric, p.nb, data.data());
    cpuIdx->add(p.nb, data.data());
    cpuIdx->nprobe = std::min((size_t)8, (size_t)p.nlist);

    std::vector<float> cpuDist(p.nq * p.k);
    std::vector<faiss::idx_t> cpuLab(p.nq * p.k);
    cpuIdx->search(p.nq, queries.data(), p.k, cpuDist.data(), cpuLab.data());

    faiss::gpu_metal::MetalIndexIVFScalarQuantizer metalIdx(
            res, cpuIdx.get());

    std::vector<float> gpuDist(p.nq * p.k);
    std::vector<faiss::idx_t> gpuLab(p.nq * p.k);
    metalIdx.search(
            p.nq, queries.data(), p.k, gpuDist.data(), gpuLab.data());

    expectRecall(p.nq, p.k, p.recallThreshold, cpuLab.data(), gpuLab.data());
}

INSTANTIATE_TEST_SUITE_P(
        SQ8_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_8bit,
                               faiss::METRIC_L2, 0.85f},
                IVFSQTestParam{128, 32, 10000, 100, 20,
                               faiss::ScalarQuantizer::QT_8bit,
                               faiss::METRIC_L2, 0.80f},
                IVFSQTestParam{32, 8, 2000, 30, 5,
                               faiss::ScalarQuantizer::QT_8bit,
                               faiss::METRIC_L2, 0.90f}));

INSTANTIATE_TEST_SUITE_P(
        SQ8_IP,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_8bit,
                               faiss::METRIC_INNER_PRODUCT, 0.85f},
                IVFSQTestParam{128, 32, 10000, 100, 20,
                               faiss::ScalarQuantizer::QT_8bit,
                               faiss::METRIC_INNER_PRODUCT, 0.80f}));

INSTANTIATE_TEST_SUITE_P(
        FP16_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_fp16,
                               faiss::METRIC_L2, 0.90f},
                IVFSQTestParam{128, 32, 10000, 100, 20,
                               faiss::ScalarQuantizer::QT_fp16,
                               faiss::METRIC_L2, 0.85f}));

INSTANTIATE_TEST_SUITE_P(
        FP16_IP,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_fp16,
                               faiss::METRIC_INNER_PRODUCT, 0.90f},
                IVFSQTestParam{128, 32, 10000, 100, 20,
                               faiss::ScalarQuantizer::QT_fp16,
                               faiss::METRIC_INNER_PRODUCT, 0.85f}));

INSTANTIATE_TEST_SUITE_P(
        SQ8Uniform_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_8bit_uniform,
                               faiss::METRIC_L2, 0.85f}));

INSTANTIATE_TEST_SUITE_P(
        SQ8Direct_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_8bit_direct,
                               faiss::METRIC_L2, 0.95f}));

INSTANTIATE_TEST_SUITE_P(
        QT4_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_4bit,
                               faiss::METRIC_L2, 0.95f}));

INSTANTIATE_TEST_SUITE_P(
        QT6_L2,
        AccMetalIndexIVFSQ,
        ::testing::Values(
                IVFSQTestParam{64, 16, 5000, 50, 10,
                               faiss::ScalarQuantizer::QT_6bit,
                               faiss::METRIC_L2, 0.95f}));

// ============================================================
//  Float16 coarse quantizer tests
// ============================================================

TEST(AccMetalIndexIVFSQ_FP16Coarse, SQ8_L2) {
    auto res = std::make_shared<faiss::gpu_metal::MetalResources>();
    if (!res->isAvailable()) {
        GTEST_SKIP() << "Metal not available";
    }

    const int dim = 64, nlist = 16, nb = 5000, nq = 50, k = 10;
    std::vector<float> data(nb * dim), queries(nq * dim);
    faiss::float_rand(data.data(), data.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    auto cpuIdx = makeCpuIVFSQ(
            dim, nlist, faiss::ScalarQuantizer::QT_8bit,
            faiss::METRIC_L2, nb, data.data());
    cpuIdx->add(nb, data.data());
    cpuIdx->nprobe = 4;

    std::vector<float> cpuD(nq * k);
    std::vector<faiss::idx_t> cpuL(nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16CoarseQuantizer = true;
    faiss::gpu_metal::MetalIndexIVFScalarQuantizer metalIdx(
            res, cpuIdx.get(), config);

    std::vector<float> gpuD(nq * k);
    std::vector<faiss::idx_t> gpuL(nq * k);
    metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());

    expectRecall(nq, k, 0.80f, cpuL.data(), gpuL.data());
}

TEST(AccMetalIndexIVFSQ_FP16Coarse, FP16_IP) {
    auto res = std::make_shared<faiss::gpu_metal::MetalResources>();
    if (!res->isAvailable()) {
        GTEST_SKIP() << "Metal not available";
    }

    const int dim = 64, nlist = 16, nb = 5000, nq = 50, k = 10;
    std::vector<float> data(nb * dim), queries(nq * dim);
    faiss::float_rand(data.data(), data.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    auto cpuIdx = makeCpuIVFSQ(
            dim, nlist, faiss::ScalarQuantizer::QT_fp16,
            faiss::METRIC_INNER_PRODUCT, nb, data.data());
    cpuIdx->add(nb, data.data());
    cpuIdx->nprobe = 4;

    std::vector<float> cpuD(nq * k);
    std::vector<faiss::idx_t> cpuL(nq * k);
    cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16CoarseQuantizer = true;
    faiss::gpu_metal::MetalIndexIVFScalarQuantizer metalIdx(
            res, cpuIdx.get(), config);

    std::vector<float> gpuD(nq * k);
    std::vector<faiss::idx_t> gpuL(nq * k);
    metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());

    expectRecall(nq, k, 0.85f, cpuL.data(), gpuL.data());
}

TEST(AccMetalIndexIVFSQ_ExtraTypes, Qt4AndQt6ConstructAndSearch) {
    auto res = std::make_shared<faiss::gpu_metal::MetalResources>();
    if (!res->isAvailable()) {
        GTEST_SKIP() << "Metal not available";
    }

    const int dim = 32, nlist = 8, nb = 2000, nq = 20, k = 8;
    std::vector<float> data(nb * dim), queries(nq * dim);
    faiss::float_rand(data.data(), data.size(), 501);
    faiss::float_rand(queries.data(), queries.size(), 502);

    for (auto qt : {faiss::ScalarQuantizer::QT_4bit, faiss::ScalarQuantizer::QT_6bit}) {
        auto cpuIdx = makeCpuIVFSQ(dim, nlist, qt, faiss::METRIC_L2, nb, data.data());
        cpuIdx->add(nb, data.data());
        cpuIdx->nprobe = 6;

        faiss::gpu_metal::MetalIndexIVFScalarQuantizer metalIdx(res, cpuIdx.get());
        EXPECT_EQ(metalIdx.sqQuantizerType(), qt);

        std::vector<float> cpuD(nq * k), gpuD(nq * k);
        std::vector<faiss::idx_t> cpuL(nq * k), gpuL(nq * k);
        cpuIdx->search(nq, queries.data(), k, cpuD.data(), cpuL.data());
        metalIdx.search(nq, queries.data(), k, gpuD.data(), gpuL.data());
        expectRecall(nq, k, 0.95f, cpuL.data(), gpuL.data());
    }
}

} // namespace
