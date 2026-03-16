// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * CPU vs Metal GPU IVFPQ benchmark (8-bit PQ, L2 metric).
 *
 * Usage:
 *   ./BenchMetalIvfPQVsCpu
 *   ./BenchMetalIvfPQVsCpu --m 8
 *   ./BenchMetalIvfPQVsCpu --m 16
 */

#import <chrono>
#import <cstdio>
#import <cstdlib>
#import <cstring>
#import <set>
#import <string>
#import <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu_metal/MetalIndexIVFPQ.h>
#include <faiss/gpu_metal/MetalResources.h>
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
            const faiss::idx_t lab = refLabels[static_cast<size_t>(q) * k + i];
            if (lab >= 0) {
                refSet.insert(lab);
            }
        }
        for (int i = 0; i < k; ++i) {
            const faiss::idx_t lab =
                    testLabels[static_cast<size_t>(q) * k + i];
            if (lab >= 0 && refSet.count(lab)) {
                ++hits;
            }
        }
        total += refSet.size();
    }
    return total ? static_cast<double>(hits) / static_cast<double>(total) : 1.0;
}

static std::unique_ptr<faiss::IndexIVFPQ> makeCpuIVFPQ(
        int d,
        int nlist,
        int M,
        int nbits,
        int nb,
        const float* xb) {
    auto* coarse = new faiss::IndexFlatL2(d);
    auto idx = std::make_unique<faiss::IndexIVFPQ>(
            coarse,
            static_cast<size_t>(d),
            static_cast<size_t>(nlist),
            static_cast<size_t>(M),
            static_cast<size_t>(nbits),
            faiss::METRIC_L2);
    idx->own_fields = true;
    idx->train(nb, xb);
    idx->add(nb, xb);
    return idx;
}

int main(int argc, char** argv) {
    int M = 16;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Usage: %s [--m <subquantizers>]\n", argv[0]);
            return 2;
        }
    }

    std::printf("=== Faiss Metal vs CPU IVFPQ Benchmark ===\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        std::printf("Metal not available. Skipping runs.\n");
        return 1;
    }

    const int d = 128;
    const int nb = 100000;
    const int nq = 1000;
    const int nlist = 4096;
    const int nbits = 8;
    const int nprobe = 32;
    const int nWarmup = 2;
    const int nRuns = 5;
    const int ks[] = {20, 50, 100, 128, 256, 512, 1024, 2048};
    const int numK = static_cast<int>(sizeof(ks) / sizeof(ks[0]));

    if (M <= 0 || (d % M) != 0) {
        std::fprintf(stderr, "Invalid M=%d for d=%d\n", M, d);
        return 2;
    }

    std::printf(
            "Config: d=%d nb=%d nq=%d nlist=%d nprobe=%d M=%d nbits=%d\n",
            d,
            nb,
            nq,
            nlist,
            nprobe,
            M,
            nbits);
    std::printf("Warmup: %d runs, Timed: %d runs\n\n", nWarmup, nRuns);

    std::vector<float> xb(static_cast<size_t>(nb) * d);
    std::vector<float> xq(static_cast<size_t>(nq) * d);
    faiss::float_rand(xb.data(), xb.size(), 12345);
    faiss::float_rand(xq.data(), xq.size(), 67890);

    auto cpuIndex = makeCpuIVFPQ(d, nlist, M, nbits, nb, xb.data());
    cpuIndex->nprobe = static_cast<size_t>(nprobe);

    faiss::gpu_metal::MetalIndexIVFPQ gpuIndex(
            metalRes->getResources(),
            cpuIndex.get());

    const double payloadMemMB = (static_cast<double>(nb) *
                                 (cpuIndex->code_size + sizeof(faiss::idx_t))) /
            (1024.0 * 1024.0);

    for (int ki = 0; ki < numK; ++ki) {
        const int k = ks[ki];
        std::printf("=== k = %d ===\n", k);

        std::vector<float> cpuDist(static_cast<size_t>(nq) * k);
        std::vector<faiss::idx_t> cpuLabels(static_cast<size_t>(nq) * k);
        std::vector<float> gpuDist(static_cast<size_t>(nq) * k);
        std::vector<faiss::idx_t> gpuLabels(static_cast<size_t>(nq) * k);

        for (int i = 0; i < nWarmup; ++i) {
            cpuIndex->search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());
            gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());
        }

        double t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            cpuIndex->search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());
        }
        const double cpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;

        t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());
        }
        const double gpuMs = (nowSeconds() - t0) * 1000.0 / nRuns;

        const double rec = recallAtK(nq, k, cpuLabels.data(), gpuLabels.data());

        std::printf("CPU IVFPQ:  %.3f ms per search (avg of %d runs)\n", cpuMs, nRuns);
        std::printf(
                "Metal IVFPQ: %.3f ms per search (avg of %d runs)\n",
                gpuMs,
                nRuns);
        if (gpuMs > 0) {
            std::printf("Speedup (CPU/Metal): %.2fx\n", cpuMs / gpuMs);
        }
        std::printf("Queries/sec CPU:  %.0f\n", nq / (cpuMs / 1000.0));
        std::printf("Queries/sec Metal: %.0f\n", nq / (gpuMs / 1000.0));
        std::printf("Recall@%d (Metal vs CPU labels): %.6f\n", k, rec);
        std::printf(
                "SUMMARY,index=IVFPQ,metric=L2,nb=%d,nq=%d,d=%d,nlist=%d,nprobe=%d,"
                "m=%d,nbits=%d,k=%d,cpu_ms=%.6f,gpu_ms=%.6f,cpu_qps=%.2f,"
                "gpu_qps=%.2f,speedup=%.6f,cpu_mem_mb=%.3f,metal_mem_mb=%.3f,"
                "recall_at_k=%.6f\n",
                nb,
                nq,
                d,
                nlist,
                nprobe,
                M,
                nbits,
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
