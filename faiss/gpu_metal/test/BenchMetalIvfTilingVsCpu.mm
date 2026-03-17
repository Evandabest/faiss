// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * CPU vs Metal GPU IVFFlat benchmark focused on query-tiling behavior under
 * constrained FAISS_METAL_IVF_QUERY_TILE_BYTES budgets.
 */

#import <chrono>
#import <cstdio>
#import <cstdlib>
#import <cstring>
#import <string>
#import <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

namespace {

double nowSeconds() {
    return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

struct ScopedEnvVar {
    std::string key;
    std::string oldVal;
    bool hadOld = false;

    ScopedEnvVar(const char* k, const char* v) : key(k ? k : "") {
        const char* old = std::getenv(key.c_str());
        if (old) {
            hadOld = true;
            oldVal = old;
        }
        setenv(key.c_str(), v, 1);
    }

    ~ScopedEnvVar() {
        if (hadOld) {
            setenv(key.c_str(), oldVal.c_str(), 1);
        } else {
            unsetenv(key.c_str());
        }
    }
};

size_t choosePredictedTileRows(
        size_t nq,
        int d,
        int k,
        size_t nprobe,
        int nlist,
        size_t budgetBytes) {
    size_t perQuery = 0;
    perQuery += (size_t)d * sizeof(float);
    perQuery += (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (size_t)k * (sizeof(float) + sizeof(int64_t));
    perQuery += nprobe * (sizeof(float) + sizeof(int32_t));
    perQuery += (size_t)nlist * sizeof(float);
    if (perQuery == 0) {
        return nq;
    }
    size_t tile = budgetBytes / perQuery;
    tile = std::max<size_t>(tile, 1);
    tile = std::min(tile, nq);
    return tile;
}

size_t getIvfFullCoarseMaxBytesBench() {
    const char* env = std::getenv("FAISS_METAL_IVF_FULL_COARSE_MAX_BYTES");
    if (!env || env[0] == '\0') {
        return 16ULL * 1024 * 1024;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(env, &end, 10);
    if (end == env) {
        return 16ULL * 1024 * 1024;
    }
    return (size_t)v;
}

bool useFullCoarseGpuForIvfBench() {
    const char* env = std::getenv("FAISS_METAL_IVF_USE_FULL_COARSE");
    if (!env || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

bool logSyncProfileForIvfBench() {
    const char* env = std::getenv("FAISS_METAL_IVF_LOG_SYNC_PROFILE");
    if (!env || env[0] == '\0') {
        return false;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' ||
        env[0] == 'F') {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    ScopedEnvVar strictFallback("FAISS_METAL_IVF_ALLOW_CPU_FALLBACK", "0");
    int d = 128;
    int nb = 100000;
    int nq = 4096;
    int nlist = 4096;
    int nprobe = 32;
    int k = 100;
    int nWarmup = 1;
    int nRuns = 3;

    std::vector<size_t> budgets = {
            16ULL * 1024 * 1024,
            32ULL * 1024 * 1024,
            64ULL * 1024 * 1024,
            128ULL * 1024 * 1024,
            256ULL * 1024 * 1024};

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            d = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            nb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            nq = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nlist") == 0 && i + 1 < argc) {
            nlist = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nprobe") == 0 && i + 1 < argc) {
            nprobe = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            nRuns = std::atoi(argv[++i]);
        }
    }

    printf("=== Faiss Metal IVFFlat Tiling Budget Benchmark ===\n\n");
    printf(
            "Config: d=%d nb=%d nq=%d nlist=%d nprobe=%d k=%d warmup=%d runs=%d\n\n",
            d,
            nb,
            nq,
            nlist,
            nprobe,
            k,
            nWarmup,
            nRuns);
    printf("Fallback policy: strict GPU mode (FAISS_METAL_IVF_ALLOW_CPU_FALLBACK=0)\n\n");
    const bool fullCoarseEnabled = useFullCoarseGpuForIvfBench();
    const size_t fullCoarseMaxBytes = getIvfFullCoarseMaxBytesBench();
    printf(
            "Coarse policy: full_matrix=%s max_bytes=%zu\n\n",
            fullCoarseEnabled ? "on" : "off",
            fullCoarseMaxBytes);
    printf(
            "Sync profile logging: %s (FAISS_METAL_IVF_LOG_SYNC_PROFILE)\n\n",
            logSyncProfileForIvfBench() ? "on" : "off");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        printf("Metal not available. Skipping runs.\n");
        return 1;
    }

    std::vector<float> db((size_t)nb * d);
    std::vector<float> queries((size_t)nq * d);
    faiss::float_rand(db.data(), db.size(), 5501);
    faiss::float_rand(queries.data(), queries.size(), 5502);

    faiss::IndexFlatL2 coarse(d);
    faiss::IndexIVFFlat cpuIndex(&coarse, d, nlist, faiss::METRIC_L2);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(nb, db.data());
    cpuIndex.add(nb, db.data());

    std::vector<float> cpuDist((size_t)nq * k);
    std::vector<faiss::idx_t> cpuLab((size_t)nq * k);

    const double t0Cpu = nowSeconds();
    for (int i = 0; i < nRuns; ++i) {
        cpuIndex.search(nq, queries.data(), k, cpuDist.data(), cpuLab.data());
    }
    const double cpuMs = (nowSeconds() - t0Cpu) * 1000.0 / nRuns;
    printf("CPU baseline: %.3f ms/search (qps=%.0f)\n\n", cpuMs, nq / (cpuMs / 1000.0));

    for (size_t budget : budgets) {
        char budgetEnv[32];
        std::snprintf(budgetEnv, sizeof(budgetEnv), "%llu", (unsigned long long)budget);
        setenv("FAISS_METAL_IVF_QUERY_TILE_BYTES", budgetEnv, 1);

        faiss::gpu_metal::MetalIndexIVFFlat metalIndex(
                metalRes->getResources(), d, nlist, faiss::METRIC_L2);
        metalIndex.train(nb, db.data());
        metalIndex.add(nb, db.data());

        faiss::IVFSearchParameters params;
        params.nprobe = nprobe;

        std::vector<float> metalDist((size_t)nq * k);
        std::vector<faiss::idx_t> metalLab((size_t)nq * k);

        for (int i = 0; i < nWarmup; ++i) {
            metalIndex.search(
                    nq, queries.data(), k, metalDist.data(), metalLab.data(), &params);
        }

        const double t0 = nowSeconds();
        for (int i = 0; i < nRuns; ++i) {
            metalIndex.search(
                    nq, queries.data(), k, metalDist.data(), metalLab.data(), &params);
        }
        const double metalMs = (nowSeconds() - t0) * 1000.0 / nRuns;
        const double speedup = (metalMs > 0.0) ? (cpuMs / metalMs) : 0.0;

        const size_t tileRows = choosePredictedTileRows(
                (size_t)nq, d, k, (size_t)nprobe, nlist, budget);
        const size_t numTiles = ((size_t)nq + tileRows - 1) / tileRows;
        const size_t coarseMatrixBytes = tileRows * (size_t)nlist * sizeof(float);
        const bool predictedFullCoarse = fullCoarseEnabled && coarseMatrixBytes <= fullCoarseMaxBytes;

        printf("Budget: %zu MB\n", budget / (1024 * 1024));
        printf(
                "  predicted tile rows: %zu (%zu tiles)\n",
                tileRows,
                numTiles);
        printf(
                "  predicted coarse path: %s (tile coarse bytes=%zu)\n",
                predictedFullCoarse ? "full_matrix" : "non_matrix",
                coarseMatrixBytes);
        printf(
                "  Metal: %.3f ms/search (qps=%.0f) speedup(CPU/Metal)=%.3fx\n",
                metalMs,
                nq / (metalMs / 1000.0),
                speedup);
        printf(
                "SUMMARY,index=IVFFlatTiling,metric=L2,d=%d,nb=%d,nq=%d,nlist=%d,nprobe=%d,k=%d,"
                "tile_budget_bytes=%zu,predicted_tile_rows=%zu,predicted_num_tiles=%zu,"
                "coarse_full_enabled=%d,coarse_full_max_bytes=%zu,coarse_matrix_bytes=%zu,"
                "predicted_full_coarse=%d,cpu_ms=%.6f,metal_ms=%.6f,speedup_cpu_over_metal=%.6f\n",
                d,
                nb,
                nq,
                nlist,
                nprobe,
                k,
                budget,
                tileRows,
                numTiles,
                fullCoarseEnabled ? 1 : 0,
                fullCoarseMaxBytes,
                coarseMatrixBytes,
                predictedFullCoarse ? 1 : 0,
                cpuMs,
                metalMs,
                speedup);
        printf("\n");
    }

    unsetenv("FAISS_METAL_IVF_QUERY_TILE_BYTES");
    printf("==================\n");
    return 0;
}
