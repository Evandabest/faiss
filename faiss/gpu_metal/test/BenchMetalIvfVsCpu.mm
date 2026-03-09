// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * CPU vs Metal GPU IVFFlat search benchmark. Sweeps over k values
 * (20, 50, 100, 128, 256, 512, 1024, 2048) and reports CPU vs Metal time and speedup per k.
 * IVF supports k up to 2048 (getMetalDistanceMaxK()).
 */

#import <chrono>
#import <cstdio>
#import <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

static double nowSecondsIvf() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Faiss Metal vs CPU IVFFlat Benchmark ===\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        printf("Metal not available. Skipping runs.\n");
        return 1;
    }

    const int d = 128;
    const int nb = 100000;
    const int nq = 1000;
    const int nlist = 4096;
    const int nprobe = 32;
    const int nWarmup = 2;
    const int nRuns = 5;

    const int ks[] = {20, 50, 100, 128, 256, 512, 1024, 2048};
    const int numK = (int)(sizeof(ks) / sizeof(ks[0]));

    printf("Config: d=%d nb=%d nq=%d nlist=%d nprobe=%d\n", d, nb, nq, nlist, nprobe);
    printf("Warmup: %d runs, Timed: %d runs\n\n", nWarmup, nRuns);

    std::vector<float> db(nb * d);
    std::vector<float> queries(nq * d);
    faiss::float_rand(db.data(), nb * d, 12345);
    faiss::float_rand(queries.data(), nq * d, 67890);

    faiss::IndexFlatL2 coarse(d);
    faiss::IndexIVFFlat cpuIndex(&coarse, d, nlist, faiss::METRIC_L2);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, db.data());
    cpuIndex.add(nb, db.data());

    auto gpuIndex = std::make_shared<faiss::gpu_metal::MetalIndexIVFFlat>(
            metalRes->getResources(), d, nlist, faiss::METRIC_L2, 0.0f);
    gpuIndex->train(nb, db.data());
    gpuIndex->add(nb, db.data());

    for (int ki = 0; ki < numK; ++ki) {
        int k = ks[ki];
        printf("=== k = %d ===\n", k);

        std::vector<float> cpuDist(nq * k);
        std::vector<faiss::idx_t> cpuLabels(nq * k);
        std::vector<float> gpuDist(nq * k);
        std::vector<faiss::idx_t> gpuLabels(nq * k);

        for (int i = 0; i < nWarmup; ++i) {
            cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
            gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
        }

        double t0 = nowSecondsIvf();
        for (int i = 0; i < nRuns; ++i)
            cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
        double cpuMs = (nowSecondsIvf() - t0) * 1000.0 / nRuns;

        t0 = nowSecondsIvf();
        for (int i = 0; i < nRuns; ++i)
            gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
        double gpuMs = (nowSecondsIvf() - t0) * 1000.0 / nRuns;

        printf("CPU IVFFlat:  %.3f ms per search (avg of %d runs)\n", cpuMs, nRuns);
        printf("Metal IVFFlat: %.3f ms per search (avg of %d runs)\n", gpuMs, nRuns);
        if (gpuMs > 0)
            printf("Speedup (CPU/Metal): %.2fx\n", cpuMs / gpuMs);
        printf("Queries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
        printf("Queries/sec Metal: %.0f\n", nq / (gpuMs / 1000.0));
        printf("\n");
    }

    printf("==================\n");
    return 0;
}

