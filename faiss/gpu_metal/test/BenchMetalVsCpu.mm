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
 *   ./BenchMetalFlatVsCpu             — full sweep (k=10,20,...,2048), CPU then GPU each
 *   ./BenchMetalFlatVsCpu <k>          — run only that k (e.g. 20), both CPU and GPU
 *   ./BenchMetalFlatVsCpu <k> gpu      — run only GPU for that k (one process, then exit)
 *   ./BenchMetalFlatVsCpu --fp16       — run Metal with float16 storage
 */

#import <chrono>
#import <cstdio>
#import <cstdlib>
#import <cstring>
#import <cstdint>
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

static bool isIntegerArg(const char* s) {
    if (!s || !*s) {
        return false;
    }
    if (*s == '-') {
        ++s;
    }
    if (!*s) {
        return false;
    }
    while (*s) {
        if (*s < '0' || *s > '9') {
            return false;
        }
        ++s;
    }
    return true;
}

static double recallAtK(
        const faiss::idx_t* cpuLabels,
        const faiss::idx_t* gpuLabels,
        int nq,
        int k) {
    size_t overlap = 0;
    const size_t total = static_cast<size_t>(nq) * static_cast<size_t>(k);
    for (int q = 0; q < nq; ++q) {
        const faiss::idx_t* cpuRow = cpuLabels + static_cast<size_t>(q) * k;
        const faiss::idx_t* gpuRow = gpuLabels + static_cast<size_t>(q) * k;
        for (int i = 0; i < k; ++i) {
            const faiss::idx_t tgt = gpuRow[i];
            for (int j = 0; j < k; ++j) {
                if (cpuRow[j] == tgt) {
                    ++overlap;
                    break;
                }
            }
        }
    }
    return total ? static_cast<double>(overlap) / static_cast<double>(total) : 0.0;
}

int main(int argc, char** argv) {
    int singleK = -1;
    bool runCpu = true;
    bool runGpu = true;
    bool useFp16 = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg) {
            continue;
        }
        if (strcmp(arg, "--fp16") == 0 || strcmp(arg, "fp16") == 0) {
            useFp16 = true;
            continue;
        }
        if (strcmp(arg, "gpu") == 0) {
            runCpu = false;
            continue;
        }
        if (strcmp(arg, "cpu") == 0) {
            runGpu = false;
            continue;
        }
        if (isIntegerArg(arg)) {
            int kArg = atoi(arg);
            if (kArg > 0) {
                singleK = kArg;
            }
            continue;
        }
        fprintf(stderr, "Unknown arg: %s\n", arg);
        fprintf(
                stderr,
                "Usage: %s [k] [cpu|gpu] [--fp16]\n",
                argv[0]);
        return 2;
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

    printf(
            "Config: d=%d nb=%d nq=%d useFp16=%s\n",
            d,
            nb,
            nq,
            useFp16 ? "true" : "false");
    printf("Warmup: %d runs, Timed: %d runs\n", nWarmup, nRuns);
    if (singleK >= 0) {
        const char* mode = runCpu && runGpu ? "CPU+GPU"
                : (runCpu ? "CPU only" : "GPU only");
        printf(">>> This process: k=%d only (%s) <<<\n", singleK, mode);
    }
    printf("\n");

    std::vector<float> db(nb * d);
    std::vector<float> queries(nq * d);
    faiss::float_rand(db.data(), nb * d, 12345);
    faiss::float_rand(queries.data(), nq * d, 67890);

    faiss::IndexFlatL2 cpuIndex(d);
    cpuIndex.add(nb, db.data());

    faiss::gpu_metal::MetalIndexConfig config;
    config.useFloat16 = useFp16;
    auto gpuIndex = std::make_shared<faiss::gpu_metal::MetalIndexFlat>(
            metalRes->getResources(),
            d,
            faiss::MetricType::METRIC_L2,
            0.0f,
            config);
    gpuIndex->add(nb, db.data());

    const double cpuMemMB = (static_cast<double>(nb) * d * sizeof(float)) /
            (1024.0 * 1024.0);
    const double metalMemMB = (static_cast<double>(nb) * d *
                               (useFp16 ? sizeof(uint16_t) : sizeof(float))) /
            (1024.0 * 1024.0);

    for (int ki = 0; ki < (int)kList.size(); ++ki) {
        int k = kList[ki];
        printf("=== k = %d ===\n", k);

        std::vector<float> cpuDist(nq * k);
        std::vector<faiss::idx_t> cpuLabels(nq * k);
        std::vector<float> gpuDist(nq * k);
        std::vector<faiss::idx_t> gpuLabels(nq * k);

        double cpuMs = 0.0, gpuMs = 0.0;

        if (runCpu) {
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

        if (runGpu) {
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

        if (runCpu && runGpu && gpuMs > 0)
            printf("Speedup (CPU/GPU): %.2fx\n", cpuMs / gpuMs);

        double rec = -1.0;
        if (runCpu && runGpu) {
            rec = recallAtK(cpuLabels.data(), gpuLabels.data(), nq, k);
            printf("Recall@%d (Metal vs CPU labels): %.6f\n", k, rec);
        }

        printf(
                "SUMMARY,index=Flat,metric=L2,use_fp16=%d,nb=%d,nq=%d,d=%d,k=%d,"
                "cpu_ms=%.6f,gpu_ms=%.6f,cpu_qps=%.2f,gpu_qps=%.2f,speedup=%.6f,"
                "cpu_mem_mb=%.3f,metal_mem_mb=%.3f,recall_at_k=%.6f\n",
                useFp16 ? 1 : 0,
                nb,
                nq,
                d,
                k,
                cpuMs,
                gpuMs,
                cpuMs > 0.0 ? (nq / (cpuMs / 1000.0)) : 0.0,
                gpuMs > 0.0 ? (nq / (gpuMs / 1000.0)) : 0.0,
                (cpuMs > 0.0 && gpuMs > 0.0) ? (cpuMs / gpuMs) : 0.0,
                cpuMemMB,
                metalMemMB,
                rec);
        printf("\n");
    }

    printf("==================\n");
    return 0;
}
