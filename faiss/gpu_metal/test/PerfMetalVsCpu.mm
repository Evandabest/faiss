// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * TEMPORARY: CPU vs Metal GPU Flat index search benchmark.
 * Run from Xcode or command line to compare times. Remove or move to perf/ later.
 */

#import <chrono>
#import <cstdio>
#import <vector>

#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

static double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Faiss Metal vs CPU Flat Search Benchmark ===\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        printf("Metal not available. Skipping GPU runs.\n");
        return 1;
    }

    // Configurable sizes (change these to test different scenarios)
    const int d = 128;
    const int nb = 100000;   // database size
    const int nq = 1000;     // number of queries
    const int nWarmup = 2;
    const int nRuns = 5;

    // Sweep over k to see behavior across small/medium/large k
    const int ks[] = {10, 20, 50, 100, 128, 256, 512};
    const int numK = (int)(sizeof(ks) / sizeof(ks[0]));

    printf("Config: d=%d nb=%d nq=%d\n", d, nb, nq);
    printf("Warmup: %d runs, Timed: %d runs\n\n", nWarmup, nRuns);

    std::vector<float> db(nb * d);
    std::vector<float> queries(nq * d);
    faiss::float_rand(db.data(), nb * d, 12345);
    faiss::float_rand(queries.data(), nq * d, 67890);

    // Build CPU index
    faiss::IndexFlatL2 cpuIndex(d);
    cpuIndex.add(nb, db.data());

    // Build Metal index
    auto gpuIndex = std::make_shared<faiss::gpu_metal::MetalIndexFlat>(
            metalRes->getResources(), d, faiss::MetricType::METRIC_L2, 0.0f);
    gpuIndex->add(nb, db.data());

    for (int ki = 0; ki < numK; ++ki) {
        int k = ks[ki];
        printf("=== k = %d ===\n", k);

        std::vector<float> cpuDist(nq * k);
        std::vector<faiss::idx_t> cpuLabels(nq * k);
        std::vector<float> gpuDist(nq * k);
        std::vector<faiss::idx_t> gpuLabels(nq * k);

        // Warmup
        for (int i = 0; i < nWarmup; ++i) {
            cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
            gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
        }

        // Time CPU
        double t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
        }
        double t1 = nowSeconds();
        double cpuMs = (t1 - t0) * 1000.0 / nRuns;

        // Time GPU
        t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
        }
        t1 = nowSeconds();
        double gpuMs = (t1 - t0) * 1000.0 / nRuns;

        printf("CPU:  %.3f ms per search (avg of %d runs)\n", cpuMs, nRuns);
        printf("Metal: %.3f ms per search (avg of %d runs)\n", gpuMs, nRuns);
        if (gpuMs > 0) {
            double speedup = cpuMs / gpuMs;
            printf("Speedup (CPU/GPU): %.2fx\n", speedup);
        }
        printf("Queries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
        printf("Queries/sec Metal: %.0f\n", nq / (gpuMs / 1000.0));
        printf("\n");
    }

    printf("==================\n");

    return 0;
}
