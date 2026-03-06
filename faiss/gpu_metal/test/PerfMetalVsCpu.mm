// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * CPU vs Metal GPU Flat index search benchmark.
 *
 * Usage:
 *   ./PerfMetalVsCpu               — full sweep (k=10,20,50,100), CPU then GPU each
 *   ./PerfMetalVsCpu <k>           — run only that k (e.g. 20), both CPU and GPU
 *   ./PerfMetalVsCpu <k> gpu       — run only GPU for that k (one process, then exit)
 */

#import <chrono>
#import <cstdio>
#import <cstdlib>
#import <string>
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
    int singleK = -1;
    bool gpuOnly = false;
    bool cpuOnly = false;
    if (argc >= 2) {
        singleK = atoi(argv[1]);
        if (singleK <= 0) singleK = -1;
    }
    if (argc >= 3) {
        std::string side(argv[2]);
        if (side == "gpu") gpuOnly = true;
        else if (side == "cpu") cpuOnly = true;
    }

    printf("=== Faiss Metal vs CPU Flat Search Benchmark ===\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        printf("Metal not available. Skipping GPU runs.\n");
        return 1;
    }

    const int d = 128;
    const int nb = 100000;
    const int nq = 1000;
    const int nWarmup = 4;
    const int nRuns = 10;

    const int ks[] = {10, 20, 50, 100, 128, 256, 512, 1024, 2048, 4096};
    const int numK = (int)(sizeof(ks) / sizeof(ks[0]));

    std::vector<int> kList;
    if (singleK >= 0) {
        kList.push_back(singleK);
    } else {
        for (int i = 0; i < numK; ++i) kList.push_back(ks[i]);
    }

    printf("Config: d=%d nb=%d nq=%d\n", d, nb, nq);
    printf("Warmup: %d runs, Timed: %d runs\n", nWarmup, nRuns);
    if (singleK >= 0) {
        printf(">>> This process: k=%d only (%s) <<<\n", singleK, gpuOnly ? "GPU only" : cpuOnly ? "CPU only" : "CPU+GPU");
    }
    printf("\n");

    std::vector<float> db(nb * d);
    std::vector<float> queries(nq * d);
    faiss::float_rand(db.data(), nb * d, 12345);
    faiss::float_rand(queries.data(), nq * d, 67890);

    faiss::IndexFlatL2 cpuIndex(d);
    cpuIndex.add(nb, db.data());

    auto gpuIndex = std::make_shared<faiss::gpu_metal::MetalIndexFlat>(
            metalRes->getResources(), d, faiss::MetricType::METRIC_L2, 0.0f);
    gpuIndex->add(nb, db.data());

    for (int ki = 0; ki < (int)kList.size(); ++ki) {
        int k = kList[ki];
        printf("=== k = %d ===\n", k);

        std::vector<float> cpuDist(nq * k);
        std::vector<faiss::idx_t> cpuLabels(nq * k);
        std::vector<float> gpuDist(nq * k);
        std::vector<faiss::idx_t> gpuLabels(nq * k);

        double cpuMs = 0.0, gpuMs = 0.0;

        if (!cpuOnly) {
            for (int i = 0; i < nWarmup; ++i)
                cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
            double t0 = nowSeconds();
            for (int i = 0; i < nRuns; ++i) {
                //printf("CPU run #: %d\n", i);
                cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
            }
            cpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;
            printf("CPU:  %.3f ms per search (avg of %d runs)\n", cpuMs, nRuns);
            printf("Queries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
        }

        if (!gpuOnly) {
            for (int i = 0; i < nWarmup; ++i)
                gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
            double t0 = nowSeconds();
            for (int i = 0; i < nRuns; ++i) {
                //printf("GPU run #: %d\n", i);
                gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
            }
            gpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;
            printf("Metal: %.3f ms per search (avg of %d runs)\n", gpuMs, nRuns);
            printf("Queries/sec Metal: %.0f\n", nq / (gpuMs / 1000.0));
        }

        if (!cpuOnly && !gpuOnly && gpuMs > 0)
            printf("Speedup (CPU/GPU): %.2fx\n", cpuMs / gpuMs);
        printf("\n");
    }

    printf("==================\n");
    return 0;
}
