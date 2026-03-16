// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * CPU vs Metal GPU BinaryFlat benchmark (Hamming).
 */

#import <chrono>
#import <cstdio>
#import <set>
#import <vector>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalIndexBinaryFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

static double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double recallAtK(
        int nq,
        int k,
        const faiss::idx_t* refLabels,
        const faiss::idx_t* testLabels) {
    size_t hits = 0;
    size_t total = 0;
    for (int q = 0; q < nq; ++q) {
        std::set<faiss::idx_t> refSet;
        for (int i = 0; i < k; ++i) {
            faiss::idx_t id = refLabels[static_cast<size_t>(q) * k + i];
            if (id >= 0) {
                refSet.insert(id);
            }
        }
        for (int i = 0; i < k; ++i) {
            faiss::idx_t id = testLabels[static_cast<size_t>(q) * k + i];
            if (id >= 0 && refSet.count(id)) {
                ++hits;
            }
        }
        total += refSet.size();
    }
    return total ? static_cast<double>(hits) / static_cast<double>(total) : 1.0;
}

int main() {
    std::printf("=== Faiss Metal vs CPU BinaryFlat Benchmark ===\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        std::printf("Metal not available. Skipping runs.\n");
        return 1;
    }

    const int d = 256; // bits
    const int codeSize = d / 8;
    const int nb = 100000;
    const int nq = 1000;
    const int nWarmup = 3;
    const int nRuns = 10;
    const int ks[] = {10, 20, 50, 100, 128, 256, 512, 1024};
    const int numK = static_cast<int>(sizeof(ks) / sizeof(ks[0]));

    std::printf("Config: d=%d(bits) nb=%d nq=%d\n", d, nb, nq);
    std::printf("Warmup: %d runs, Timed: %d runs\n\n", nWarmup, nRuns);

    std::vector<uint8_t> xb(static_cast<size_t>(nb) * codeSize);
    std::vector<uint8_t> xq(static_cast<size_t>(nq) * codeSize);
    faiss::byte_rand(xb.data(), xb.size(), 12345);
    faiss::byte_rand(xq.data(), xq.size(), 67890);

    faiss::IndexBinaryFlat cpuIndex(d);
    cpuIndex.add(nb, xb.data());

    faiss::gpu_metal::MetalIndexBinaryFlat gpuIndex(
            metalRes->getResources(),
            &cpuIndex);

    const double payloadMemMB = (static_cast<double>(nb) * codeSize) /
            (1024.0 * 1024.0);

    for (int ki = 0; ki < numK; ++ki) {
        const int k = ks[ki];
        std::printf("=== k = %d ===\n", k);

        std::vector<int32_t> cpuDist(static_cast<size_t>(nq) * k);
        std::vector<faiss::idx_t> cpuLabels(static_cast<size_t>(nq) * k);
        std::vector<int32_t> gpuDist(static_cast<size_t>(nq) * k);
        std::vector<faiss::idx_t> gpuLabels(static_cast<size_t>(nq) * k);

        for (int i = 0; i < nWarmup; ++i) {
            cpuIndex.search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());
            gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());
        }

        double t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            cpuIndex.search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());
        }
        const double cpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;

        t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());
        }
        const double gpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;

        const double rec = recallAtK(nq, k, cpuLabels.data(), gpuLabels.data());

        std::printf(
                "CPU BinaryFlat:  %.3f ms per search (avg of %d runs)\n",
                cpuMs,
                nRuns);
        std::printf(
                "Metal BinaryFlat: %.3f ms per search (avg of %d runs)\n",
                gpuMs,
                nRuns);
        if (gpuMs > 0.0) {
            std::printf("Speedup (CPU/Metal): %.2fx\n", cpuMs / gpuMs);
        }
        std::printf("Queries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
        std::printf("Queries/sec Metal: %.0f\n", nq / (gpuMs / 1000.0));
        std::printf("Recall@%d (Metal vs CPU labels): %.6f\n", k, rec);
        std::printf(
                "SUMMARY,index=BinaryFlat,metric=Hamming,nb=%d,nq=%d,d=%d,k=%d,"
                "cpu_ms=%.6f,gpu_ms=%.6f,cpu_qps=%.2f,gpu_qps=%.2f,speedup=%.6f,"
                "cpu_mem_mb=%.3f,metal_mem_mb=%.3f,recall_at_k=%.6f\n",
                nb,
                nq,
                d,
                k,
                cpuMs,
                gpuMs,
                cpuMs > 0.0 ? (nq / (cpuMs / 1000.0)) : 0.0,
                gpuMs > 0.0 ? (nq / (gpuMs / 1000.0)) : 0.0,
                (cpuMs > 0.0 && gpuMs > 0.0) ? (cpuMs / gpuMs) : 0.0,
                payloadMemMB,
                payloadMemMB,
                rec);
        std::printf("\n");
    }

    std::printf("==================\n");
    return 0;
}
