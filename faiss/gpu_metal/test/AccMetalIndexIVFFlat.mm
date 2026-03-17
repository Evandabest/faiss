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
#include <algorithm>
#include <cstdlib>
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

/// Assert exact label match and near-exact distance match element-wise.
void expectExactResults(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab,
        float atol = 1e-5f) {
    for (int i = 0; i < nq * k; ++i) {
        EXPECT_EQ(refLab[i], testLab[i]) << "label mismatch at i=" << i;
        if (refLab[i] < 0) {
            continue;
        }
        EXPECT_NEAR(refDist[i], testDist[i], atol) << "distance mismatch at i=" << i;
    }
}

/// Assert exact distances while allowing permutations only inside tie groups.
void expectExactResultsAllowingTiePermutations(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab,
        float atol = 1e-5f,
        float tieTol = 1e-5f) {
    for (int q = 0; q < nq; ++q) {
        int i = 0;
        while (i < k) {
            int base = q * k;
            int ii = base + i;

            if (refLab[ii] == testLab[ii]) {
                if (refLab[ii] >= 0) {
                    EXPECT_NEAR(refDist[ii], testDist[ii], atol)
                            << "distance mismatch at q=" << q << " i=" << i;
                }
                ++i;
                continue;
            }

            const float refAnchor = refDist[ii];
            const float testAnchor = testDist[ii];
            int refEnd = i + 1;
            int testEnd = i + 1;
            while (refEnd < k &&
                   std::fabs(refDist[base + refEnd] - refAnchor) <= tieTol) {
                ++refEnd;
            }
            while (testEnd < k &&
                   std::fabs(testDist[base + testEnd] - testAnchor) <= tieTol) {
                ++testEnd;
            }

            ASSERT_EQ(refEnd - i, testEnd - i)
                    << "tie-group size mismatch at q=" << q << " i=" << i;

            std::vector<faiss::idx_t> refGroup;
            std::vector<faiss::idx_t> testGroup;
            refGroup.reserve((size_t)(refEnd - i));
            testGroup.reserve((size_t)(testEnd - i));

            for (int j = i; j < refEnd; ++j) {
                const int pos = base + j;
                if (refLab[pos] >= 0) {
                    EXPECT_NEAR(refDist[pos], testDist[pos], tieTol)
                            << "tie-group distance mismatch at q=" << q
                            << " i=" << j;
                }
                refGroup.push_back(refLab[pos]);
                testGroup.push_back(testLab[pos]);
            }

            std::sort(refGroup.begin(), refGroup.end());
            std::sort(testGroup.begin(), testGroup.end());
            EXPECT_EQ(refGroup, testGroup)
                    << "tie-group label-set mismatch at q=" << q
                    << " start_i=" << i;

            i = refEnd;
        }
    }
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

TEST_F(AccMetalIndexIVFFlat, ExactL2SingleListAdversarialMatchesCpu) {
    const int dim = 24, nb = 3000, nq = 16, nlist = 1, k = 64;
    const size_t nprobe = 1;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 2021);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 2022);

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

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactIPFullProbeMatchesCpu) {
    const int dim = 32, nb = 4096, nq = 20, nlist = 16, k = 40;
    const size_t nprobe = 16; // full probe

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 2031);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 2032);

    auto cpuIdx =
            makeCpuIVFFlat(dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_INNER_PRODUCT);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, L2_SmallQueryTileBudgetMatchesCpu) {
    const int dim = 64, nb = 5000, nq = 192, nlist = 32, k = 10;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 2011);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 2012);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    setenv("FAISS_METAL_IVF_QUERY_TILE_BYTES", "1048576", 1); // 1 MB

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    unsetenv("FAISS_METAL_IVF_QUERY_TILE_BYTES");
    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
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

TEST_F(AccMetalIndexIVFFlat, RejectsOutOfRangePreassignedListIds) {
    const int dim = 32, nb = 1024, nq = 4, nlist = 16, k = 4;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8111);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8112);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> distances((size_t)nq * k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
    std::vector<faiss::idx_t> assign((size_t)nq * (size_t)nprobe, 0);
    std::vector<float> centroidDistances((size_t)nq * (size_t)nprobe, 0.0f);

    assign[0] = (faiss::idx_t)std::numeric_limits<int32_t>::max() + 1;

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

TEST_F(AccMetalIndexIVFFlat, SearchPreassignedRejectsStorePairs) {
    const int dim = 32, nb = 512, nq = 2, nlist = 8, k = 4;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8121);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8122);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> distances((size_t)nq * k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
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
            true,
            &params,
            nullptr));
}

TEST_F(AccMetalIndexIVFFlat, SearchPreassignedRejectsStats) {
    const int dim = 32, nb = 512, nq = 2, nlist = 8, k = 4;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8131);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8132);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> distances((size_t)nq * k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
    std::vector<faiss::idx_t> assign((size_t)nq * (size_t)nprobe, 0);
    std::vector<float> centroidDistances((size_t)nq * (size_t)nprobe, 0.0f);

    faiss::IVFSearchParameters params;
    params.nprobe = nprobe;
    faiss::IndexIVFStats stats;

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
            &stats));
}

TEST_F(AccMetalIndexIVFFlat, SearchPreassignedRejectsMaxCodes) {
    const int dim = 32, nb = 512, nq = 2, nlist = 8, k = 4;
    const int nprobe = 4;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8141);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8142);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> distances((size_t)nq * k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
    std::vector<faiss::idx_t> assign((size_t)nq * (size_t)nprobe, 0);
    std::vector<float> centroidDistances((size_t)nq * (size_t)nprobe, 0.0f);

    faiss::IVFSearchParameters params;
    params.nprobe = nprobe;
    params.max_codes = 123;

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

TEST_F(AccMetalIndexIVFFlat, ExactSearchPreassignedMatchesCpu) {
    const int dim = 40, nb = 6000, nq = 12, nlist = 32, k = 24;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8149);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8150);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> centroidDist((size_t)nq * nprobe);
    std::vector<faiss::idx_t> assign((size_t)nq * nprobe);
    cpuIdx->quantizer->search(
            nq, queries.data(), (faiss::idx_t)nprobe, centroidDist.data(), assign.data());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    cpuIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            refD.data(),
            refL.data(),
            false,
            &ivfParams,
            nullptr);
    metalIdx.search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            testD.data(),
            testL.data(),
            false,
            &ivfParams,
            nullptr);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactSearchPreassignedChunkedListMatchesCpu) {
    // Force preassigned list-chunking: one inverted list with len > 1024.
    const int dim = 32, nb = 7000, nq = 10, nlist = 1, k = 64;
    const size_t nprobe = 1;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8181);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8182);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> centroidDist((size_t)nq * nprobe, 0.0f);
    std::vector<faiss::idx_t> assign((size_t)nq * nprobe, 0);
    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    cpuIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            refD.data(),
            refL.data(),
            false,
            &ivfParams,
            nullptr);
    metalIdx.search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            testD.data(),
            testL.data(),
            false,
            &ivfParams,
            nullptr);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactSearchPreassignedInterleavedMatchesCpu) {
    const int dim = 40, nb = 6000, nq = 12, nlist = 32, k = 24;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8159);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8160);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = true;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIdx = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIdx, nullptr);
    EXPECT_TRUE(metalIdx->interleavedLayout());

    std::vector<float> centroidDist((size_t)nq * nprobe);
    std::vector<faiss::idx_t> assign((size_t)nq * nprobe);
    cpuIdx->quantizer->search(
            nq, queries.data(), (faiss::idx_t)nprobe, centroidDist.data(), assign.data());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    cpuIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            refD.data(),
            refL.data(),
            false,
            &ivfParams,
            nullptr);
    metalIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            testD.data(),
            testL.data(),
            false,
            &ivfParams,
            nullptr);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ExactSearchPreassignedInterleavedIPMatchesCpu) {
    const int dim = 40, nb = 6000, nq = 12, nlist = 32, k = 24;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8169);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8170);

    auto cpuIdx =
            makeCpuIVFFlat(dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = true;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIdx = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIdx, nullptr);
    EXPECT_TRUE(metalIdx->interleavedLayout());

    std::vector<float> centroidDist((size_t)nq * nprobe);
    std::vector<faiss::idx_t> assign((size_t)nq * nprobe);
    cpuIdx->quantizer->search(
            nq, queries.data(), (faiss::idx_t)nprobe, centroidDist.data(), assign.data());

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    cpuIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            refD.data(),
            refL.data(),
            false,
            &ivfParams,
            nullptr);
    metalIdx->search_preassigned(
            nq,
            queries.data(),
            k,
            assign.data(),
            centroidDist.data(),
            testD.data(),
            testL.data(),
            false,
            &ivfParams,
            nullptr);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, StrictFallbackModeRejectsSearchFallback) {
    // Trigger a known GPU envelope rejection regardless of scan chunking policy.
    const int dim = 513, nb = 8000, nq = 8, nlist = 64, k = 32;
    const size_t nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8151);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8152);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> d((size_t)nq * k);
    std::vector<faiss::idx_t> l((size_t)nq * k, -1);

    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    EXPECT_ANY_THROW(metalIdx.search(
            nq, queries.data(), k, d.data(), l.data(), &ivfParams));
    unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
}

TEST_F(AccMetalIndexIVFFlat, StrictFallbackModeRejectsPreassignedFallback) {
    // Trigger a known GPU envelope rejection regardless of scan chunking policy.
    const int dim = 513, nb = 8000, nq = 4, nlist = 64, k = 32;
    const int nprobe = 8;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8161);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 8162);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    std::vector<float> distances((size_t)nq * k, 0.0f);
    std::vector<faiss::idx_t> labels((size_t)nq * k, -1);
    std::vector<faiss::idx_t> assign((size_t)nq * (size_t)nprobe, 0);
    std::vector<float> centroidDistances((size_t)nq * (size_t)nprobe, 0.0f);
    for (int q = 0; q < nq; ++q) {
        for (int p = 0; p < nprobe; ++p) {
            assign[(size_t)q * (size_t)nprobe + (size_t)p] = p;
        }
    }

    faiss::IVFSearchParameters params;
    params.nprobe = nprobe;

    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
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
    unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
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

TEST_F(AccMetalIndexIVFFlat, IncrementalAppendInBatchesMatchesCpu) {
    const int dim = 48, nb = 4096, nq = 32, nlist = 32, k = 12;
    const size_t nprobe = 10;
    const int batch = 257; // uneven batch to stress per-list growth

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 5501);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5502);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());

    for (int off = 0; off < nb; off += batch) {
        int cur = std::min(batch, nb - off);
        cpuIdx->add(cur, vecs.data() + (size_t)off * dim);
        metalIdx.add(cur, vecs.data() + (size_t)off * dim);
    }

    EXPECT_EQ(metalIdx.ntotal, nb);

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
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

TEST_F(AccMetalIndexIVFFlat, ReserveMemoryThenBatchedAddMatchesCpu) {
    const int dim = 40, nb = 4096, nq = 24, nlist = 32, k = 10;
    const size_t nprobe = 8;
    const int batch = 173;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 5601);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(queries.data(), queries.size(), 5602);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.reserveMemory(nb);

    for (int off = 0; off < nb; off += batch) {
        int cur = std::min(batch, nb - off);
        cpuIdx->add(cur, vecs.data() + (size_t)off * dim);
        metalIdx.add(cur, vecs.data() + (size_t)off * dim);
    }

    EXPECT_EQ(metalIdx.ntotal, nb);

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectRecall(nq, k, 0.85f, refL.data(), testL.data());
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
    const size_t nprobe = 64; // nprobe * k = 2048, covered by chunked-probe GPU scan

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
    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    EXPECT_NO_THROW(metalRaw->search(nq, queries.data(), k, testD.data(), testL.data()));
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

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
    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    EXPECT_NO_THROW(metalRaw->search(nq, queries.data(), k, testD.data(), testL.data()));
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

    expectRecall(nq, k, 1.0f, refL.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ExactL2MultiListBoundaryMatchesCpu) {
    const int dim = 64, nb = 4096, nq = 20, nlist = 64, k = 128;
    const size_t nprobe = 8; // nprobe * k = 1024 exactness boundary

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 4601);
    faiss::float_rand(queries.data(), queries.size(), 4602);

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

    expectExactResultsAllowingTiePermutations(
            nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactIPMultiListBoundaryMatchesCpu) {
    const int dim = 64, nb = 4096, nq = 20, nlist = 64, k = 128;
    const size_t nprobe = 8; // nprobe * k = 1024 exactness boundary

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 4701);
    faiss::float_rand(queries.data(), queries.size(), 4702);

    auto cpuIdx =
            makeCpuIVFFlat(dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_INNER_PRODUCT);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalIdx.search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactL2InterleavedBoundaryMatchesCpu) {
    const int dim = 64, nb = 4096, nq = 20, nlist = 64, k = 128;
    const size_t nprobe = 8; // nprobe * k = 1024 exactness boundary

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 4801);
    faiss::float_rand(queries.data(), queries.size(), 4802);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->nprobe = nprobe;
    cpuIdx->add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.interleavedLayout = true;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);
    auto* metalIVF = dynamic_cast<faiss::gpu_metal::MetalIndexIVFFlat*>(metalRaw);
    ASSERT_NE(metalIVF, nullptr);
    EXPECT_TRUE(metalIVF->interleavedLayout());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * k), testD((size_t)nq * k);
    std::vector<faiss::idx_t> refL((size_t)nq * k, -1), testL((size_t)nq * k, -1);

    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());
    metalRaw->search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams);

    expectExactResults(nq, k, refD.data(), refL.data(), testD.data(), testL.data());

    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ScaleStressMatrixMatchesCpuRecall) {
    struct StressCase {
        int d;
        int k;
        size_t nprobe;
        faiss::MetricType metric;
        bool skewed;
        float minRecall;
        int seed;
    };

    const std::vector<StressCase> cases = {
            {32, 1, 1, faiss::METRIC_L2, false, 0.99f, 9101},
            {32, 32, 8, faiss::METRIC_L2, false, 0.92f, 9102},
            {64, 64, 16, faiss::METRIC_L2, false, 0.88f, 9103},
            {128, 32, 16, faiss::METRIC_L2, false, 0.82f, 9104},
            {128, 32, 16, faiss::METRIC_L2, true, 0.78f, 9105},
            {32, 32, 8, faiss::METRIC_INNER_PRODUCT, false, 0.90f, 9106},
            {64, 64, 16, faiss::METRIC_INNER_PRODUCT, false, 0.82f, 9107},
            {128, 32, 16, faiss::METRIC_INNER_PRODUCT, true, 0.72f, 9108},
    };

    const int nlist = 64;
    const int nb = 12000;
    const int nq = 24;

    for (const auto& c : cases) {
        ASSERT_LE((size_t)c.k * c.nprobe, (size_t)1024)
                << "Scale stress accuracy matrix only gates regimes inside the "
                << "current validated nprobe*k envelope";
        SCOPED_TRACE(testing::Message() << "d=" << c.d << " k=" << c.k
                                        << " nprobe=" << c.nprobe
                                        << " metric="
                                        << (c.metric == faiss::METRIC_L2 ? "L2" : "IP")
                                        << " skewed=" << c.skewed);

        std::vector<float> vecs((size_t)nb * c.d);
        std::vector<float> queries((size_t)nq * c.d);
        faiss::float_rand(vecs.data(), vecs.size(), c.seed);
        faiss::float_rand(queries.data(), queries.size(), c.seed + 1000);

        if (c.skewed) {
            // Force a large duplicate-like cluster to stress skewed list behavior.
            for (int i = 0; i < nb / 2; ++i) {
                for (int j = 0; j < c.d; ++j) {
                    vecs[(size_t)i * c.d + j] = vecs[j] + 0.0005f * (float)(i % 17);
                }
            }
        }

        auto cpuIdx = makeCpuIVFFlat(c.d, nlist, c.metric, nb, vecs.data());
        cpuIdx->add(nb, vecs.data());
        cpuIdx->nprobe = c.nprobe;

        faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
                resources_, c.d, (faiss::idx_t)nlist, c.metric);
        metalIdx.train(nb, vecs.data());
        metalIdx.add(nb, vecs.data());

        faiss::IVFSearchParameters ivfParams;
        ivfParams.nprobe = c.nprobe;

        std::vector<float> refD((size_t)nq * (size_t)c.k), testD((size_t)nq * (size_t)c.k);
        std::vector<faiss::idx_t> refL((size_t)nq * (size_t)c.k, -1),
                testL((size_t)nq * (size_t)c.k, -1);
        cpuIdx->search(nq, queries.data(), c.k, refD.data(), refL.data());
        metalIdx.search(
                nq,
                queries.data(),
                c.k,
                testD.data(),
                testL.data(),
                &ivfParams);

        expectRecall(nq, c.k, c.minRecall, refL.data(), testL.data());
    }
}

TEST_F(AccMetalIndexIVFFlat, HighPressureStrictModeMatrixMatchesCpuRecall) {
    struct StressCase {
        int d;
        int nlist;
        int nb;
        int nq;
        int k;
        size_t nprobe;
        faiss::MetricType metric;
        bool skewed;
        float minRecall;
        int seed;
    };

    const std::vector<StressCase> cases = {
            // Probe-chunked regime: nprobe * k > 1024.
            {32, 64, 8000, 24, 32, 64, faiss::METRIC_L2, false, 1.00f, 9201},
            {32, 64, 8000, 24, 32, 64, faiss::METRIC_INNER_PRODUCT, false, 0.99f, 9202},
            // List-chunked regime: single list with len > 1024.
            {24, 1, 7000, 20, 64, 1, faiss::METRIC_L2, false, 1.00f, 9203},
            // Combined pressure with skewed data.
            {24, 1, 7000, 20, 64, 1, faiss::METRIC_L2, true, 0.99f, 9204},
    };

    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);

    for (const auto& c : cases) {
        SCOPED_TRACE(testing::Message()
                     << "d=" << c.d << " nlist=" << c.nlist << " nb=" << c.nb
                     << " nq=" << c.nq << " k=" << c.k << " nprobe=" << c.nprobe
                     << " metric="
                     << (c.metric == faiss::METRIC_L2 ? "L2" : "IP")
                     << " skewed=" << c.skewed);

        std::vector<float> vecs((size_t)c.nb * c.d);
        std::vector<float> queries((size_t)c.nq * c.d);
        faiss::float_rand(vecs.data(), vecs.size(), c.seed);
        faiss::float_rand(queries.data(), queries.size(), c.seed + 1000);

        if (c.skewed) {
            for (int i = 0; i < c.nb / 2; ++i) {
                for (int j = 0; j < c.d; ++j) {
                    vecs[(size_t)i * c.d + j] = vecs[j] + 0.0005f * (float)(i % 17);
                }
            }
        }

        auto cpuIdx = makeCpuIVFFlat(c.d, c.nlist, c.metric, c.nb, vecs.data());
        cpuIdx->add(c.nb, vecs.data());
        cpuIdx->nprobe = c.nprobe;

        faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
                resources_, c.d, (faiss::idx_t)c.nlist, c.metric);
        metalIdx.train(c.nb, vecs.data());
        metalIdx.add(c.nb, vecs.data());

        faiss::IVFSearchParameters ivfParams;
        ivfParams.nprobe = c.nprobe;

        std::vector<float> refD((size_t)c.nq * (size_t)c.k), testD((size_t)c.nq * (size_t)c.k);
        std::vector<faiss::idx_t> refL((size_t)c.nq * (size_t)c.k, -1),
                testL((size_t)c.nq * (size_t)c.k, -1);
        cpuIdx->search(c.nq, queries.data(), c.k, refD.data(), refL.data());
        EXPECT_NO_THROW(metalIdx.search(
                c.nq,
                queries.data(),
                c.k,
                testD.data(),
                testL.data(),
                &ivfParams));
        expectRecall(c.nq, c.k, c.minRecall, refL.data(), testL.data());
    }

    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }
}

TEST_F(AccMetalIndexIVFFlat, ForcedChunkedSelectionMatchesDefault) {
    const int dim = 64, nlist = 64, nb = 8000, nq = 24, k = 32;
    const size_t nprobe = 16;

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9301);
    faiss::float_rand(queries.data(), queries.size(), 9302);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> dDefault((size_t)nq * k), dForced((size_t)nq * k);
    std::vector<faiss::idx_t> lDefault((size_t)nq * k, -1), lForced((size_t)nq * k, -1);

    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    const char* oldForceEnv = std::getenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION");
    const bool hadOldForce = oldForceEnv != nullptr;
    std::string oldForce = oldForceEnv ? oldForceEnv : "";

    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    unsetenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION");
    metalIdx.search(nq, queries.data(), k, dDefault.data(), lDefault.data(), &ivfParams);

    setenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION", "1", 1);
    metalIdx.search(nq, queries.data(), k, dForced.data(), lForced.data(), &ivfParams);

    if (hadOldForce) {
        setenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION", oldForce.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_FORCE_CHUNKED_SELECTION");
    }
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

    expectExactResultsAllowingTiePermutations(
            nq, k, dDefault.data(), lDefault.data(), dForced.data(), lForced.data());
}

TEST_F(AccMetalIndexIVFFlat, ReducedExactCandidateBudgetMatchesDefault) {
    const int dim = 64, nlist = 64, nb = 8000, nq = 24, k = 32;
    const size_t nprobe = 16;

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9331);
    faiss::float_rand(queries.data(), queries.size(), 9332);

    faiss::gpu_metal::MetalIndexIVFFlat metalIdx(
            resources_, dim, (faiss::idx_t)nlist, faiss::METRIC_L2);
    metalIdx.train(nb, vecs.data());
    metalIdx.add(nb, vecs.data());

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> dDefault((size_t)nq * k), dBudget((size_t)nq * k);
    std::vector<faiss::idx_t> lDefault((size_t)nq * k, -1), lBudget((size_t)nq * k, -1);

    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    const char* oldBudgetEnv = std::getenv("FAISS_METAL_IVF_EXACT_CANDIDATES");
    const bool hadOldBudget = oldBudgetEnv != nullptr;
    std::string oldBudget = oldBudgetEnv ? oldBudgetEnv : "";

    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    unsetenv("FAISS_METAL_IVF_EXACT_CANDIDATES");
    metalIdx.search(nq, queries.data(), k, dDefault.data(), lDefault.data(), &ivfParams);

    setenv("FAISS_METAL_IVF_EXACT_CANDIDATES", "256", 1);
    metalIdx.search(nq, queries.data(), k, dBudget.data(), lBudget.data(), &ivfParams);

    if (hadOldBudget) {
        setenv("FAISS_METAL_IVF_EXACT_CANDIDATES", oldBudget.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_EXACT_CANDIDATES");
    }
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

    expectExactResultsAllowingTiePermutations(
            nq, k, dDefault.data(), lDefault.data(), dBudget.data(), lBudget.data());
}

TEST_F(AccMetalIndexIVFFlat, ExactL2HighEnvelopePressureMatchesCpu) {
    // Stress selection envelope with large k and high nprobe. This forces
    // probe chunking while requiring strict-mode GPU execution.
    const int dim = 64, nlist = 128, nb = 24000, nq = 16, k = 256;
    const size_t nprobe = 32; // nprobe * k = 8192

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9311);
    faiss::float_rand(queries.data(), queries.size(), 9312);

    auto cpuIdx = makeCpuIVFFlat(dim, nlist, faiss::METRIC_L2, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * (size_t)k), testD((size_t)nq * (size_t)k);
    std::vector<faiss::idx_t> refL((size_t)nq * (size_t)k, -1),
            testL((size_t)nq * (size_t)k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());

    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    EXPECT_NO_THROW(
            metalRaw->search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams));
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

    expectRecall(nq, k, 1.0f, refL.data(), testL.data());
    delete metalRaw;
}

TEST_F(AccMetalIndexIVFFlat, ExactIPHighEnvelopePressureMatchesCpu) {
    // Mirror the L2 high-envelope stress under inner product metric.
    const int dim = 64, nlist = 128, nb = 24000, nq = 16, k = 256;
    const size_t nprobe = 32; // nprobe * k = 8192

    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9321);
    faiss::float_rand(queries.data(), queries.size(), 9322);

    auto cpuIdx = makeCpuIVFFlat(
            dim, nlist, faiss::METRIC_INNER_PRODUCT, nb, vecs.data());
    cpuIdx->add(nb, vecs.data());
    cpuIdx->nprobe = nprobe;

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, cpuIdx.get(), &opts);
    ASSERT_NE(metalRaw, nullptr);

    faiss::IVFSearchParameters ivfParams;
    ivfParams.nprobe = nprobe;

    std::vector<float> refD((size_t)nq * (size_t)k), testD((size_t)nq * (size_t)k);
    std::vector<faiss::idx_t> refL((size_t)nq * (size_t)k, -1),
            testL((size_t)nq * (size_t)k, -1);
    cpuIdx->search(nq, queries.data(), k, refD.data(), refL.data());

    const char* oldFallbackEnv = std::getenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    const bool hadOldFallback = oldFallbackEnv != nullptr;
    std::string oldFallback = oldFallbackEnv ? oldFallbackEnv : "";
    setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0", 1);
    EXPECT_NO_THROW(
            metalRaw->search(nq, queries.data(), k, testD.data(), testL.data(), &ivfParams));
    if (hadOldFallback) {
        setenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", oldFallback.c_str(), 1);
    } else {
        unsetenv("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK");
    }

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
