// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * TEMPORARY: CPU vs Metal GPU IVFFlat search benchmark.
 * NOTE: Current MetalIndexIVFFlat is CPU-backed; this is mainly a
 * sanity/perf placeholder for when IVF is moved to GPU.
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

    printf("=== Faiss Metal vs CPU IVFFlat Benchmark (Metal IVF is CPU-backed for now) ===\n\n");

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
    const int k = 20;

    printf("Config: d=%d nb=%d nq=%d nlist=%d nprobe=%d k=%d\n", d, nb, nq, nlist, nprobe, k);
    printf("Warmup: %d runs, Timed: %d runs\n\n", nWarmup, nRuns);

    std::vector<float> db(nb * d);
    std::vector<float> queries(nq * d);
    faiss::float_rand(db.data(), nb * d, 12345);
    faiss::float_rand(queries.data(), nq * d, 67890);

    // CPU IVFFlat (L2)
    faiss::IndexFlatL2 coarse(d);
    faiss::IndexIVFFlat cpuIndex(&coarse, d, nlist, faiss::METRIC_L2);
    cpuIndex.nprobe = nprobe;

    // Train and add on CPU
    cpuIndex.train(nb, db.data());
    cpuIndex.add(nb, db.data());

    // Metal IVFFlat wrapper (currently CPU-backed)
    auto gpuIndex = std::make_shared<faiss::gpu_metal::MetalIndexIVFFlat>(
            metalRes->getResources(), d, nlist, faiss::METRIC_L2, 0.0f);
    gpuIndex->train(nb, db.data());
    gpuIndex->add(nb, db.data());

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
    double t0 = nowSecondsIvf();
    for (int i = 0; i < nRuns; ++i) {
        cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLabels.data());
    }
    double t1 = nowSecondsIvf();
    double cpuMs = (t1 - t0) * 1000.0 / nRuns;

    // Time Metal (currently CPU-backed IVFFlat wrapper)
    t0 = nowSecondsIvf();
    for (int i = 0; i < nRuns; ++i) {
        gpuIndex->search(nq, queries.data(), k, gpuDist.data(), gpuLabels.data());
    }
    t1 = nowSecondsIvf();
    double gpuMs = (t1 - t0) * 1000.0 / nRuns;

    printf("--- Results (IVFFlat) ---\n");
    printf("CPU IVFFlat:  %.3f ms per search (avg of %d runs)\n", cpuMs, nRuns);
    printf("MetalIndexIVFFlat (CPU-backed): %.3f ms per search (avg of %d runs)\n", gpuMs, nRuns);
    if (gpuMs > 0) {
        double speedup = cpuMs / gpuMs;
        printf("Speedup (CPU/Metal-IVF wrapper): %.2fx\n", speedup);
    }
    printf("\nQueries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
    printf("Queries/sec Metal-IVF wrapper: %.0f\n", nq / (gpuMs / 1000.0));
    printf("==================\n");

    return 0;
}

