// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Minimal C++ test for MetalIndexFlat: add, search, reset; compare to CPU IndexFlat.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalCloner.h>
#include <faiss/gpu_metal/MetalDistance.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/random.h>
#include <faiss/utils/fp16.h>
#include <gtest/gtest.h>
#include <algorithm>
#import <cmath>
#import <memory>
#import <set>
#import <vector>

namespace {

constexpr float kTolerance = 1e-5f;

void compareSearchResults(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab) {
    for (int i = 0; i < nq * k; ++i) {
        EXPECT_NEAR(refDist[i], testDist[i], kTolerance * (std::fabs(refDist[i]) + 1.0f))
                << "i=" << i;
        EXPECT_EQ(refLab[i], testLab[i]) << "i=" << i;
    }
}

// Compare results allowing tie-breaking / fp order: same set of (dist, label) per query; distances with tolerance.
void compareSearchResultsAllowTieBreak(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab) {
    std::vector<std::pair<float, faiss::idx_t>> refPairs((size_t)k), testPairs((size_t)k);
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i) {
            refPairs[i] = {refDist[q * k + i], refLab[q * k + i]};
            testPairs[i] = {testDist[q * k + i], testLab[q * k + i]};
        }
        std::sort(refPairs.begin(), refPairs.end());
        std::sort(testPairs.begin(), testPairs.end());
        for (int i = 0; i < k; ++i) {
            EXPECT_NEAR(refPairs[i].first, testPairs[i].first,
                        kTolerance * (std::fabs(refPairs[i].first) + 1.0f))
                    << "q=" << q << " i=" << i;
            EXPECT_EQ(refPairs[i].second, testPairs[i].second) << "q=" << q << " i=" << i;
        }
    }
}

// Compare results when both row and column tiling: require same set of labels per query and distances match (with tolerance).
void compareSearchResultsTiled(
        int nq,
        int k,
        const float* refDist,
        const faiss::idx_t* refLab,
        const float* testDist,
        const faiss::idx_t* testLab) {
    std::vector<faiss::idx_t> refLabSorted((size_t)k), testLabSorted((size_t)k);
    std::vector<float> refDistSorted((size_t)k), testDistSorted((size_t)k);
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i) {
            refLabSorted[i] = refLab[q * k + i];
            refDistSorted[i] = refDist[q * k + i];
            testLabSorted[i] = testLab[q * k + i];
            testDistSorted[i] = testDist[q * k + i];
        }
        std::sort(refLabSorted.begin(), refLabSorted.end());
        std::sort(testLabSorted.begin(), testLabSorted.end());
        std::sort(refDistSorted.begin(), refDistSorted.end());
        std::sort(testDistSorted.begin(), testDistSorted.end());
        for (int i = 0; i < k; ++i) {
            EXPECT_EQ(refLabSorted[i], testLabSorted[i]) << "q=" << q << " i=" << i << " (set of labels must match)";
            EXPECT_NEAR(refDistSorted[i], testDistSorted[i],
                        kTolerance * (std::fabs(refDistSorted[i]) + 1.0f))
                    << "q=" << q << " i=" << i;
        }
    }
}

} // namespace

class AccMetalIndexFlat : public ::testing::Test {
protected:
    void SetUp() override {
        resources_ = std::make_shared<faiss::gpu_metal::MetalResources>();
        if (!resources_->isAvailable()) {
            GTEST_SKIP() << "Metal not available (no device or queue)";
        }
    }
    std::shared_ptr<faiss::gpu_metal::MetalResources> resources_;
};

TEST_F(AccMetalIndexFlat, L2_AddAndSearch) {
    const int dim = 4;
    const int numVecs = 50;
    const int numQuery = 5;
    const int k = 3;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, L2_LargeK) {
    // Exercise large-k top-k variant (k=512 -> topk_threadgroup_512)
    const int dim = 32;
    const int numVecs = 600;
    const int numQuery = 4;
    const int k = 512;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 12345);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 67890);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, L2_MaxK) {
    // Exercise largest variant (k=2048 -> topk_threadgroup_2048)
    const int dim = 16;
    const int numVecs = 2500;
    const int numQuery = 2;
    const int k = 2048;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 11111);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 22222);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

// --- Tiling tests (Step 6): force two-level tiling path ---
// chooseTileSize uses ~256MB element budget; for d>32, preferredTileRows=512 so tileCols=131072.
// So nb > 131072 triggers vector tiling; nq > 512 triggers query tiling.

TEST_F(AccMetalIndexFlat, L2_TiledManyVectors) {
    // Force vector tiling: nb > tileCols (131072) -> multiple column tiles, merge path
    const int dim = 64;
    const int numVecs = 132000;  // > 131072
    const int numQuery = 20;
    const int k = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 4001);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 4002);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, L2_TiledManyQueries) {
    // Force query tiling: nq > 512 -> multiple row tiles
    const int dim = 64;
    const int numVecs = 500;
    const int numQuery = 600;  // > 512
    const int k = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 5001);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 5002);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, L2_TiledPatterns) {
    // Additional correctness coverage for tiled path with structured distance patterns.
    // Use dim=64 and nb>tileCols to force vector-tiling (like L2_TiledManyVectors).
    const int dim = 64;
    const int numVecs = 132000;  // > 131072 → vector tiling
    const int numQuery = 4;
    const int k = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    std::vector<float> queries((size_t)numQuery * dim);

    auto runPattern = [&](const char* patternName,
                          auto fillFn) {
        (void)patternName;

        fillFn(vecs, queries);

        faiss::IndexFlatL2 cpuIndex(dim);
        faiss::gpu_metal::MetalIndexFlat metalIndex(
                resources_, dim, faiss::METRIC_L2, 0.0f);
        cpuIndex.add(numVecs, vecs.data());
        metalIndex.add(numVecs, vecs.data());

        std::vector<float> refDist((size_t)numQuery * k);
        std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
        std::vector<float> testDist((size_t)numQuery * k);
        std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

        cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
        metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

        compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
    };

    // Pattern 1: strictly increasing distances w.r.t. query 0.
    runPattern("increasing", [&](auto& db, auto& qs) {
        // Query at 0; database values 0,1,2,... replicated across dim.
        for (int q = 0; q < numQuery; ++q) {
            for (int d = 0; d < dim; ++d) {
                qs[q * dim + d] = 0.0f;
            }
        }
        for (int i = 0; i < numVecs; ++i) {
            float v = (float)i;
            for (int d = 0; d < dim; ++d) {
                db[(size_t)i * dim + d] = v;
            }
        }
    });

    // Pattern 2: strictly decreasing distances w.r.t. query 0.
    runPattern("decreasing", [&](auto& db, auto& qs) {
        for (int q = 0; q < numQuery; ++q) {
            for (int d = 0; d < dim; ++d) {
                qs[q * dim + d] = 0.0f;
            }
        }
        for (int i = 0; i < numVecs; ++i) {
            float v = (float)(numVecs - 1 - i);
            for (int d = 0; d < dim; ++d) {
                db[(size_t)i * dim + d] = v;
            }
        }
    });

    // Pattern 3: many ties (all distances equal).
    runPattern("ties", [&](auto& db, auto& qs) {
        for (int q = 0; q < numQuery; ++q) {
            for (int d = 0; d < dim; ++d) {
                qs[q * dim + d] = 1.0f;
            }
        }
        for (int i = 0; i < numVecs; ++i) {
            // All vectors identical → equal distances; tie-breaking must be deterministic.
            for (int d = 0; d < dim; ++d) {
                db[(size_t)i * dim + d] = 0.0f;
            }
        }
    });
}

TEST_F(AccMetalIndexFlat, L2_TiledBoth) {
    // Force both query and vector tiling
    const int dim = 64;
    const int numVecs = 132000;
    const int numQuery = 600;
    const int k = 32;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 6001);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 6002);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    // Tiled both: require same set of labels and same set of distances per query (order may differ)
    compareSearchResultsTiled(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, L2_SingleColumnTile) {
    // One column tile (numColTiles==1) but multiple row tiles -> exercises copy path, no merge
    const int dim = 64;
    const int numVecs = 50000;   // < 131072 -> single column tile
    const int numQuery = 600;    // > 512 -> two row tiles
    const int k = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 7001);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 7002);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, IP_TiledManyVectors) {
    // Inner product with vector tiling
    const int dim = 64;
    const int numVecs = 132000;
    const int numQuery = 20;
    const int k = 10;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 8001);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 8002);

    faiss::IndexFlatIP cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_INNER_PRODUCT, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, IP_AddAndSearch) {
    const int dim = 4;
    const int numVecs = 50;
    const int numQuery = 5;
    const int k = 3;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1234);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 5678);

    faiss::IndexFlatIP cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_INNER_PRODUCT, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add(numVecs, vecs.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());
}

TEST_F(AccMetalIndexFlat, AddWithIds) {
    const int dim = 4;
    const int numVecs = 20;
    const int numQuery = 3;
    const int k = 2;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    std::vector<faiss::idx_t> ids(numVecs);
    for (int i = 0; i < numVecs; ++i) {
        ids[i] = 1000 + (faiss::idx_t)i;
    }
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 43);

    faiss::IndexFlatL2 cpuIndex(dim);
    faiss::gpu_metal::MetalIndexFlat metalIndex(
            resources_, dim, faiss::METRIC_L2, 0.0f);
    cpuIndex.add(numVecs, vecs.data());
    metalIndex.add_with_ids(numVecs, vecs.data(), ids.data());

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);

    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex.search(numQuery, queries.data(), k, testDist.data(), testLab.data());

    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_NEAR(refDist[i], testDist[i], kTolerance * (std::fabs(refDist[i]) + 1.0f));
        EXPECT_EQ(testLab[i], ids[refLab[i]]) << "Metal should return stored ids";
    }
}

TEST_F(AccMetalIndexFlat, Reset) {
    const int dim = 4;
    const int numVecs = 10;
    const int numQuery = 2;
    const int k = 1;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 100);

    faiss::gpu_metal::MetalIndexFlat index(resources_, dim, faiss::METRIC_L2, 0.0f);
    index.add(numVecs, vecs.data());
    EXPECT_EQ(index.ntotal, numVecs);

    index.reset();
    EXPECT_EQ(index.ntotal, 0);

    std::vector<float> dists((size_t)numQuery * k);
    std::vector<faiss::idx_t> labels((size_t)numQuery * k, -2);
    index.search(numQuery, queries.data(), k, dists.data(), labels.data());
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(labels[i], -1) << "after reset, labels should be -1";
    }
}

TEST_F(AccMetalIndexFlat, EmptySearch) {
    const int dim = 4;
    const int numQuery = 2;
    const int k = 1;

    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 101);

    faiss::gpu_metal::MetalIndexFlat index(resources_, dim, faiss::METRIC_L2, 0.0f);
    std::vector<float> dists((size_t)numQuery * k);
    std::vector<faiss::idx_t> labels((size_t)numQuery * k, -2);
    index.search(numQuery, queries.data(), k, dists.data(), labels.data());
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(labels[i], -1);
    }
}

TEST_F(AccMetalIndexFlat, GetNumGpus) {
    int n = faiss::gpu_metal::get_num_gpus();
    EXPECT_GE(n, 0);
    EXPECT_LE(n, 1);
    if (resources_->isAvailable()) {
        EXPECT_EQ(n, 1);
    }
}

TEST_F(AccMetalIndexFlat, IndexCpuToMetalGpu) {
    const int dim = 4;
    const int numVecs = 30;
    const int numQuery = 3;
    const int k = 2;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 200);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 201);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(numVecs, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalIndex = faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, &cpuIndex);
    ASSERT_NE(metalIndex, nullptr);
    EXPECT_EQ(metalIndex->ntotal, numVecs);

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);
    cpuIndex.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    metalIndex->search(numQuery, queries.data(), k, testDist.data(), testLab.data());
    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());

    delete metalIndex;
}

TEST_F(AccMetalIndexFlat, IndexMetalGpuToCpu) {
    const int dim = 4;
    const int numVecs = 20;
    const int numQuery = 2;
    const int k = 2;

    std::vector<float> vecs((size_t)numVecs * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 300);
    std::vector<float> queries((size_t)numQuery * dim);
    faiss::float_rand(queries.data(), queries.size(), 301);

    faiss::IndexFlatL2 cpuOrig(dim);
    cpuOrig.add(numVecs, vecs.data());

    faiss::gpu_metal::StandardMetalResources res;
    faiss::Index* metalIndex = faiss::gpu_metal::index_cpu_to_metal_gpu(&res, 0, &cpuOrig);
    ASSERT_NE(metalIndex, nullptr);
    faiss::Index* cpuBack = faiss::gpu_metal::index_metal_gpu_to_cpu(metalIndex);
    ASSERT_NE(cpuBack, nullptr);
    EXPECT_EQ(cpuBack->ntotal, numVecs);

    std::vector<float> refDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> refLab((size_t)numQuery * k, -1);
    std::vector<float> testDist((size_t)numQuery * k);
    std::vector<faiss::idx_t> testLab((size_t)numQuery * k, -1);
    cpuOrig.search(numQuery, queries.data(), k, refDist.data(), refLab.data());
    cpuBack->search(numQuery, queries.data(), k, testDist.data(), testLab.data());
    compareSearchResults(numQuery, k, refDist.data(), refLab.data(), testDist.data(), testLab.data());

    delete cpuBack;
    delete metalIndex;
}

// ============================================================
//  Float16 storage tests
// ============================================================

float computeRecall(
        int nq, int k,
        const faiss::idx_t* refLab,
        const faiss::idx_t* testLab) {
    int hits = 0, total = 0;
    for (int q = 0; q < nq; ++q) {
        std::set<faiss::idx_t> refSet;
        for (int i = 0; i < k; ++i) {
            if (refLab[q * k + i] >= 0) refSet.insert(refLab[q * k + i]);
        }
        for (int i = 0; i < k; ++i) {
            if (testLab[q * k + i] >= 0 && refSet.count(testLab[q * k + i]))
                hits++;
        }
        total += (int)refSet.size();
    }
    return total > 0 ? (float)hits / (float)total : 1.0f;
}

TEST_F(AccMetalIndexFlat, Float16L2Basic) {
    const int dim = 64;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());
    std::vector<float> cpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> cpuLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2, 0.0f, config);
    metalIdx.add(nb, vecs.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), gpuLab.data());
    EXPECT_GE(recall, 0.95f) << "Float16 L2 recall " << recall << " below 0.95";
}

TEST_F(AccMetalIndexFlat, Float16IPBasic) {
    const int dim = 64;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    faiss::IndexFlatIP cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());
    std::vector<float> cpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> cpuLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_INNER_PRODUCT, 0.0f, config);
    metalIdx.add(nb, vecs.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), gpuLab.data());
    EXPECT_GE(recall, 0.95f) << "Float16 IP recall " << recall << " below 0.95";
}

TEST_F(AccMetalIndexFlat, Float16Reconstruct) {
    const int dim = 32;
    const int nb = 100;

    std::vector<float> vecs((size_t)nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2, 0.0f, config);
    metalIdx.add(nb, vecs.data());

    std::vector<float> recons((size_t)nb * dim);
    metalIdx.reconstruct_n(0, nb, recons.data());

    for (int i = 0; i < nb * dim; ++i) {
        EXPECT_NEAR(vecs[i], recons[i], 0.01f)
                << "Float16 reconstruct mismatch at i=" << i;
    }
}

TEST_F(AccMetalIndexFlat, Float16CopyFromTo) {
    const int dim = 32;
    const int nb = 100;
    const int nq = 10;
    const int k = 5;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    faiss::IndexFlatL2 cpuOrig(dim);
    cpuOrig.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2, 0.0f, config);
    metalIdx.copyFrom(&cpuOrig);
    EXPECT_EQ(metalIdx.ntotal, nb);

    faiss::IndexFlatL2 cpuBack(dim);
    metalIdx.copyTo(&cpuBack);
    EXPECT_EQ(cpuBack.ntotal, nb);

    std::vector<float> origDist((size_t)nq * k), backDist((size_t)nq * k);
    std::vector<faiss::idx_t> origLab((size_t)nq * k), backLab((size_t)nq * k);
    cpuOrig.search(nq, queries.data(), k, origDist.data(), origLab.data());
    cpuBack.search(nq, queries.data(), k, backDist.data(), backLab.data());

    float recall = computeRecall(nq, k, origLab.data(), backLab.data());
    EXPECT_GE(recall, 0.95f) << "Float16 copyFrom/To round-trip recall " << recall;
}

TEST_F(AccMetalIndexFlat, Float16LargeD128) {
    const int dim = 128;
    const int nb = 10000;
    const int nq = 100;
    const int k = 20;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 1337);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());
    std::vector<float> cpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> cpuLab((size_t)nq * k);
    cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = true;
    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2, 0.0f, config);
    metalIdx.add(nb, vecs.data());

    std::vector<float> gpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> gpuLab((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, gpuDist.data(), gpuLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), gpuLab.data());
    EXPECT_GE(recall, 0.95f) << "Float16 L2 d=128 recall " << recall;
}

// ---- assign / compute_residual tests ----

TEST_F(AccMetalIndexFlat, AssignL2) {
    const int dim = 32, nb = 500, nq = 10, k = 3;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 42);
    faiss::float_rand(queries.data(), queries.size(), 123);

    faiss::IndexFlatL2 cpuIdx(dim);
    cpuIdx.add(nb, vecs.data());

    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2);
    metalIdx.add(nb, vecs.data());

    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);
    cpuIdx.assign(nq, queries.data(), cpuLab.data(), k);
    metalIdx.assign(nq, queries.data(), metalLab.data(), k);

    for (int i = 0; i < nq; ++i) {
        EXPECT_EQ(cpuLab[i * k], metalLab[i * k])
                << "Top-1 assign mismatch for query " << i;
    }
}

TEST_F(AccMetalIndexFlat, ComputeResidual) {
    const int dim = 32, nb = 100;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 55);

    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2);
    metalIdx.add(nb, vecs.data());

    for (int key = 0; key < nb; key += 10) {
        std::vector<float> residual(dim);
        metalIdx.compute_residual(vecs.data() + key * dim,
                                   residual.data(), key);
        for (int j = 0; j < dim; ++j) {
            EXPECT_NEAR(residual[j], 0.0f, 1e-5f)
                    << "Residual should be ~0 for own vector, key=" << key;
        }
    }

    std::vector<float> query(dim);
    faiss::float_rand(query.data(), query.size(), 77);
    std::vector<float> residual(dim), recons(dim);
    metalIdx.compute_residual(query.data(), residual.data(), 0);
    metalIdx.reconstruct(0, recons.data());
    for (int j = 0; j < dim; ++j) {
        EXPECT_NEAR(residual[j], query[j] - recons[j], 1e-5f);
    }
}

TEST_F(AccMetalIndexFlat, ComputeResidualN) {
    const int dim = 16, nb = 50, n = 5;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 99);

    faiss::gpu_metal::MetalIndexFlat metalIdx(
            resources_, dim, faiss::METRIC_L2);
    metalIdx.add(nb, vecs.data());

    std::vector<faiss::idx_t> keys = {0, 5, 10, 20, 49};
    std::vector<float> xs(n * dim), residuals(n * dim);
    for (int i = 0; i < n; ++i)
        std::memcpy(xs.data() + i * dim,
                     vecs.data() + keys[i] * dim, dim * sizeof(float));

    metalIdx.compute_residual_n(n, xs.data(), residuals.data(), keys.data());
    for (int i = 0; i < n * dim; ++i) {
        EXPECT_NEAR(residuals[i], 0.0f, 1e-5f);
    }
}

TEST_F(AccMetalIndexFlat, ClonerOptionsFloat16) {
    const int dim = 64, nb = 500, nq = 10, k = 5;
    std::vector<float> vecs(nb * dim), queries(nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 100);
    faiss::float_rand(queries.data(), queries.size(), 200);

    faiss::IndexFlatL2 cpuIdx(dim);
    cpuIdx.add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.useFloat16 = true;
    opts.verbose = true;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, &cpuIdx, &opts);
    ASSERT_NE(metalRaw, nullptr);
    EXPECT_TRUE(metalRaw->verbose);

    std::vector<float> cpuDist(nq * k), metalDist(nq * k);
    std::vector<faiss::idx_t> cpuLab(nq * k), metalLab(nq * k);
    cpuIdx.search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());
    metalRaw->search(nq, queries.data(), k, metalDist.data(), metalLab.data());

    float recall = computeRecall(nq, k, cpuLab.data(), metalLab.data());
    EXPECT_GE(recall, 0.90f) << "fp16 cloner recall = " << recall;

    delete metalRaw;
}

TEST_F(AccMetalIndexFlat, ClonerOptionsVerbose) {
    const int dim = 32, nb = 100;
    std::vector<float> vecs(nb * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 300);

    faiss::IndexFlatL2 cpuIdx(dim);
    cpuIdx.add(nb, vecs.data());

    faiss::gpu_metal::StandardMetalResources stdRes;
    faiss::gpu_metal::MetalClonerOptions opts;
    opts.verbose = false;

    faiss::Index* metalRaw = faiss::gpu_metal::index_cpu_to_metal_gpu(
            &stdRes, 0, &cpuIdx, &opts);
    ASSERT_NE(metalRaw, nullptr);
    EXPECT_FALSE(metalRaw->verbose);

    delete metalRaw;
}

TEST_F(AccMetalIndexFlat, TempPoolCachesSearchScratchBuffers) {
    const size_t poolBudget = 64ULL * 1024 * 1024;
    resources_->setTempMemoryPoolBytes(poolBudget);
    resources_->clearTempMemoryPool();
    EXPECT_EQ(resources_->getTempMemoryCachedBytes(), 0);

    const int dim = 64, nb = 8000, nq = 64, k = 20;
    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 123);
    faiss::float_rand(queries.data(), queries.size(), 456);

    faiss::gpu_metal::MetalIndexFlat metalIdx(resources_, dim, faiss::METRIC_L2);
    metalIdx.add(nb, vecs.data());

    std::vector<float> dists((size_t)nq * k);
    std::vector<faiss::idx_t> labels((size_t)nq * k);

    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());
    const size_t cachedAfterFirst = resources_->getTempMemoryCachedBytes();
    EXPECT_GT(cachedAfterFirst, 0);
    EXPECT_LE(cachedAfterFirst, poolBudget);

    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());
    const size_t cachedAfterSecond = resources_->getTempMemoryCachedBytes();
    EXPECT_EQ(cachedAfterSecond, cachedAfterFirst);
}

TEST_F(AccMetalIndexFlat, TempPoolClearAndShrinkBudget) {
    resources_->setTempMemoryPoolBytes(64ULL * 1024 * 1024);
    resources_->clearTempMemoryPool();

    const int dim = 32, nb = 6000, nq = 64, k = 10;
    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 321);
    faiss::float_rand(queries.data(), queries.size(), 654);

    faiss::gpu_metal::MetalIndexFlat metalIdx(resources_, dim, faiss::METRIC_L2);
    metalIdx.add(nb, vecs.data());

    std::vector<float> dists((size_t)nq * k);
    std::vector<faiss::idx_t> labels((size_t)nq * k);
    metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());

    const size_t cached = resources_->getTempMemoryCachedBytes();
    EXPECT_GT(cached, 0);

    const size_t smallerBudget = 2ULL * 1024 * 1024;
    resources_->setTempMemoryPoolBytes(smallerBudget);
    EXPECT_LE(resources_->getTempMemoryCachedBytes(), smallerBudget);
    EXPECT_EQ(resources_->getTempMemoryPoolBytes(), smallerBudget);

    resources_->clearTempMemoryPool();
    EXPECT_EQ(resources_->getTempMemoryCachedBytes(), 0);
}

TEST_F(AccMetalIndexFlat, MemoryInfoAndLoggingControls) {
    resources_->setTempMemoryPoolBytes(64ULL * 1024 * 1024);
    resources_->clearTempMemoryPool();
    resources_->setLogMemoryAllocations(false);
    EXPECT_FALSE(resources_->getLogMemoryAllocations());

    const int dim = 32, nb = 4000, nq = 32, k = 10;
    std::vector<float> vecs((size_t)nb * dim), queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 9001);
    faiss::float_rand(queries.data(), queries.size(), 9002);

    {
        faiss::gpu_metal::MetalIndexFlat metalIdx(
                resources_, dim, faiss::METRIC_L2);
        metalIdx.add(nb, vecs.data());

        std::vector<float> dists((size_t)nq * k);
        std::vector<faiss::idx_t> labels((size_t)nq * k);
        metalIdx.search(nq, queries.data(), k, dists.data(), labels.data());

        auto info = resources_->getMemoryInfo();
        EXPECT_GT(info.tempPoolCachedBytes, 0);
        EXPECT_FALSE(info.logMemoryAllocations);
        EXPECT_GE(info.totalLiveAllocs, (size_t)1); // FlatData is live

        const auto itTemp = info.byAllocType.find(
                (int)faiss::gpu_metal::MetalAllocType::TemporaryMemoryBuffer);
        ASSERT_NE(itTemp, info.byAllocType.end());
        EXPECT_GT(itTemp->second.totalAllocs, 0);
    }

    auto infoAfterDestroy = resources_->getMemoryInfo();
    EXPECT_EQ(infoAfterDestroy.totalLiveAllocs, 0);

    resources_->setLogMemoryAllocations(true);
    EXPECT_TRUE(resources_->getLogMemoryAllocations());
    auto infoLogging = resources_->getMemoryInfo();
    EXPECT_TRUE(infoLogging.logMemoryAllocations);
}

TEST_F(AccMetalIndexFlat, BfKnnTilingMatchesCpuL2) {
    const int dim = 64;
    const int nb = 14000;
    const int nq = 120;
    const int k = 10;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 101);
    faiss::float_rand(queries.data(), queries.size(), 202);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    const size_t vectorsMemoryLimit = (size_t)dim * sizeof(float) * 2000;
    const size_t queriesMemoryLimit =
            ((size_t)dim * sizeof(float) + (size_t)k * (sizeof(float) + sizeof(faiss::idx_t))) * 40;

    faiss::gpu_metal::bfKnn_tiling(
            resources_,
            vecs.data(),
            nb,
            queries.data(),
            nq,
            dim,
            k,
            faiss::METRIC_L2,
            testDist.data(),
            testLab.data(),
            vectorsMemoryLimit,
            queriesMemoryLimit);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsTilingI32MatchesCpuL2) {
    const int dim = 64;
    const int nb = 12000;
    const int nq = 96;
    const int k = 10;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 303);
    faiss::float_rand(queries.data(), queries.size(), 404);

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<int32_t> testLabI32((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecs.data();
    args.numVectors = nb;
    args.queries = queries.data();
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I32;
    args.outIndices = testLabI32.data();

    const size_t vectorsMemoryLimit = (size_t)dim * sizeof(float) * 1800;
    const size_t queriesMemoryLimit =
            ((size_t)dim * sizeof(float) + (size_t)k * (sizeof(float) + sizeof(int32_t))) * 32;

    faiss::gpu_metal::bfKnn_tiling(
            resources_,
            args,
            vectorsMemoryLimit,
            queriesMemoryLimit);

    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);
    for (size_t i = 0; i < testLab.size(); ++i) {
        testLab[i] = (faiss::idx_t)testLabI32[i];
    }

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsF16MatchesQuantizedCpuL2) {
    const int dim = 48;
    const int nb = 9000;
    const int nq = 64;
    const int k = 8;

    std::vector<float> vecsF32((size_t)nb * dim);
    std::vector<float> queriesF32((size_t)nq * dim);
    faiss::float_rand(vecsF32.data(), vecsF32.size(), 505);
    faiss::float_rand(queriesF32.data(), queriesF32.size(), 606);

    std::vector<uint16_t> vecsF16((size_t)nb * dim);
    std::vector<uint16_t> queriesF16((size_t)nq * dim);
    std::vector<float> vecsQuantF32((size_t)nb * dim);
    std::vector<float> queriesQuantF32((size_t)nq * dim);
    for (size_t i = 0; i < vecsF32.size(); ++i) {
        vecsF16[i] = faiss::encode_fp16(vecsF32[i]);
        vecsQuantF32[i] = faiss::decode_fp16(vecsF16[i]);
    }
    for (size_t i = 0; i < queriesF32.size(); ++i) {
        queriesF16[i] = faiss::encode_fp16(queriesF32[i]);
        queriesQuantF32[i] = faiss::decode_fp16(queriesF16[i]);
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecsQuantF32.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queriesQuantF32.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecsF16.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F16;
    args.numVectors = nb;
    args.queries = queriesF16.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F16;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    faiss::gpu_metal::bfKnn(resources_, args);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsBF16MatchesQuantizedCpuL2) {
    const int dim = 48;
    const int nb = 7000;
    const int nq = 56;
    const int k = 8;

    std::vector<float> vecsF32((size_t)nb * dim);
    std::vector<float> queriesF32((size_t)nq * dim);
    faiss::float_rand(vecsF32.data(), vecsF32.size(), 707);
    faiss::float_rand(queriesF32.data(), queriesF32.size(), 808);

    std::vector<uint16_t> vecsBF16((size_t)nb * dim);
    std::vector<uint16_t> queriesBF16((size_t)nq * dim);
    std::vector<float> vecsQuantF32((size_t)nb * dim);
    std::vector<float> queriesQuantF32((size_t)nq * dim);
    for (size_t i = 0; i < vecsF32.size(); ++i) {
        vecsBF16[i] = faiss::encode_bf16(vecsF32[i]);
        vecsQuantF32[i] = faiss::decode_bf16(vecsBF16[i]);
    }
    for (size_t i = 0; i < queriesF32.size(); ++i) {
        queriesBF16[i] = faiss::encode_bf16(queriesF32[i]);
        queriesQuantF32[i] = faiss::decode_bf16(queriesBF16[i]);
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecsQuantF32.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queriesQuantF32.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecsBF16.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::BF16;
    args.numVectors = nb;
    args.queries = queriesBF16.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::BF16;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    faiss::gpu_metal::bfKnn(resources_, args);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsColumnMajorF32MatchesCpuL2) {
    const int dim = 40;
    const int nb = 6000;
    const int nq = 48;
    const int k = 8;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 909);
    faiss::float_rand(queries.data(), queries.size(), 1001);

    std::vector<float> vecsCol((size_t)nb * dim);
    std::vector<float> queriesCol((size_t)nq * dim);
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < dim; ++j) {
            vecsCol[(size_t)j * nb + i] = vecs[(size_t)i * dim + j];
        }
    }
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < dim; ++j) {
            queriesCol[(size_t)j * nq + i] = queries[(size_t)i * dim + j];
        }
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecsCol.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.vectorsRowMajor = false;
    args.numVectors = nb;
    args.queries = queriesCol.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = false;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    const size_t vectorsMemoryLimit = (size_t)dim * sizeof(float) * 1400;
    const size_t queriesMemoryLimit =
            ((size_t)dim * sizeof(float) + (size_t)k * (sizeof(float) + sizeof(faiss::idx_t))) * 20;

    faiss::gpu_metal::bfKnn_tiling(
            resources_,
            args,
            vectorsMemoryLimit,
            queriesMemoryLimit);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsVectorNormsMatchesCpuL2) {
    const int dim = 32;
    const int nb = 5000;
    const int nq = 40;
    const int k = 7;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1102);
    faiss::float_rand(queries.data(), queries.size(), 1203);

    std::vector<float> vecNorms(nb, 0.0f);
    for (int i = 0; i < nb; ++i) {
        float n = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float v = vecs[(size_t)i * dim + j];
            n += v * v;
        }
        vecNorms[i] = n;
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecs.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.vectorsRowMajor = true;
    args.numVectors = nb;
    args.vectorNorms = vecNorms.data();
    args.queries = queries.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = true;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    const size_t vectorsMemoryLimit = (size_t)dim * sizeof(float) * 1000;
    const size_t queriesMemoryLimit =
            ((size_t)dim * sizeof(float) + (size_t)k * (sizeof(float) + sizeof(faiss::idx_t))) * 16;

    faiss::gpu_metal::bfKnn_tiling(
            resources_,
            args,
            vectorsMemoryLimit,
            queriesMemoryLimit);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsAllPairsKMinusOneMatchesCpuL2) {
    const int dim = 24;
    const int nb = 1800;
    const int nq = 36;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1304);
    faiss::float_rand(queries.data(), queries.size(), 1405);

    std::vector<float> ref((size_t)nq * nb, 0.0f);
    for (int qi = 0; qi < nq; ++qi) {
        for (int vi = 0; vi < nb; ++vi) {
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float diff = queries[(size_t)qi * dim + d] - vecs[(size_t)vi * dim + d];
                acc += diff * diff;
            }
            ref[(size_t)qi * nb + vi] = acc;
        }
    }

    std::vector<float> test((size_t)nq * nb, 0.0f);
    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = -1;
    args.dims = dim;
    args.vectors = vecs.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.vectorsRowMajor = true;
    args.numVectors = nb;
    args.queries = queries.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = true;
    args.numQueries = nq;
    args.outDistances = test.data();
    args.outIndices = nullptr;

    const size_t vectorsMemoryLimit = (size_t)dim * sizeof(float) * 700;
    const size_t queriesMemoryLimit = (size_t)dim * sizeof(float) * 18;

    faiss::gpu_metal::bfKnn_tiling(
            resources_,
            args,
            vectorsMemoryLimit,
            queriesMemoryLimit);

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(ref[i], test[i], kTolerance * (std::fabs(ref[i]) + 1.0f)) << "i=" << i;
    }
}

TEST_F(AccMetalIndexFlat, BfKnnParamsL1MetricMatchesCpu) {
    const int dim = 20;
    const int nb = 2200;
    const int nq = 30;
    const int k = 6;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1506);
    faiss::float_rand(queries.data(), queries.size(), 1607);

    faiss::IndexFlat cpuIndex(dim, faiss::METRIC_L1);
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L1;
    args.k = k;
    args.dims = dim;
    args.vectors = vecs.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.vectorsRowMajor = true;
    args.numVectors = nb;
    args.queries = queries.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = true;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    faiss::gpu_metal::bfKnn(resources_, args);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsLpMetricMatchesCpu) {
    const int dim = 18;
    const int nb = 1200;
    const int nq = 25;
    const int k = 5;
    const float p = 3.0f;

    std::vector<float> vecs((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecs.data(), vecs.size(), 1708);
    faiss::float_rand(queries.data(), queries.size(), 1809);

    faiss::IndexFlat cpuIndex(dim, faiss::METRIC_Lp);
    cpuIndex.metric_arg = p;
    cpuIndex.add(nb, vecs.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_Lp;
    args.metricArg = p;
    args.k = k;
    args.dims = dim;
    args.vectors = vecs.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.vectorsRowMajor = true;
    args.numVectors = nb;
    args.queries = queries.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = true;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    faiss::gpu_metal::bfKnn(resources_, args);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}

TEST_F(AccMetalIndexFlat, BfKnnParamsVectorNormsF16VectorsMatchesCpuL2) {
    const int dim = 16;
    const int nb = 1000;
    const int nq = 20;
    const int k = 5;

    std::vector<float> vecsF32((size_t)nb * dim);
    std::vector<float> queries((size_t)nq * dim);
    faiss::float_rand(vecsF32.data(), vecsF32.size(), 1910);
    faiss::float_rand(queries.data(), queries.size(), 2011);

    std::vector<uint16_t> vecsF16((size_t)nb * dim);
    std::vector<float> vecsQuant((size_t)nb * dim);
    for (size_t i = 0; i < vecsF32.size(); ++i) {
        vecsF16[i] = faiss::encode_fp16(vecsF32[i]);
        vecsQuant[i] = faiss::decode_fp16(vecsF16[i]);
    }
    std::vector<float> vecNorms(nb, 0.0f);
    for (int i = 0; i < nb; ++i) {
        float n = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float v = vecsQuant[(size_t)i * dim + j];
            n += v * v;
        }
        vecNorms[i] = n;
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    cpuIndex.add(nb, vecsQuant.data());

    std::vector<float> refDist((size_t)nq * k);
    std::vector<faiss::idx_t> refLab((size_t)nq * k, -1);
    cpuIndex.search(nq, queries.data(), k, refDist.data(), refLab.data());

    std::vector<float> testDist((size_t)nq * k);
    std::vector<faiss::idx_t> testLab((size_t)nq * k, -1);

    faiss::gpu_metal::MetalDistanceParams args;
    args.metric = faiss::METRIC_L2;
    args.k = k;
    args.dims = dim;
    args.vectors = vecsF16.data();
    args.vectorType = faiss::gpu_metal::MetalDistanceDataType::F16;
    args.vectorsRowMajor = true;
    args.numVectors = nb;
    args.vectorNorms = vecNorms.data();
    args.queries = queries.data();
    args.queryType = faiss::gpu_metal::MetalDistanceDataType::F32;
    args.queriesRowMajor = true;
    args.numQueries = nq;
    args.outDistances = testDist.data();
    args.outIndicesType = faiss::gpu_metal::MetalIndicesDataType::I64;
    args.outIndices = testLab.data();

    faiss::gpu_metal::bfKnn(resources_, args);

    compareSearchResultsAllowTieBreak(
            nq,
            k,
            refDist.data(),
            refLab.data(),
            testDist.data(),
            testLab.data());
}
