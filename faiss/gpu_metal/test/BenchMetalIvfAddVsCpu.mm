// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * CPU vs Metal GPU IVFFlat append benchmark.
 *
 * Measures add-path throughput for different batch sizes and reports whether
 * per-batch latency grows with database size (last quartile vs first quartile).
 */

#import <chrono>
#import <cstdlib>
#import <cstdio>
#import <cstring>
#import <memory>
#import <numeric>
#import <string>
#import <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu_metal/MetalIndexIVFFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

namespace {

double nowSec() {
    return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

struct AppendStats {
    double totalMs = 0.0;
    double avgBatchMs = 0.0;
    double firstQuartileMs = 0.0;
    double lastQuartileMs = 0.0;
    double lastOverFirst = 0.0;
    double vecsPerSec = 0.0;
    int numBatches = 0;
};

std::vector<int> parseIntList(const char* s) {
    std::vector<int> out;
    if (!s || s[0] == '\0') {
        return out;
    }
    std::string str(s);
    size_t start = 0;
    while (start < str.size()) {
        size_t comma = str.find(',', start);
        std::string tok =
                (comma == std::string::npos) ? str.substr(start) : str.substr(start, comma - start);
        if (!tok.empty()) {
            int v = std::atoi(tok.c_str());
            if (v >= 0) {
                out.push_back(v);
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

AppendStats summarizeBatchTimes(
        const std::vector<double>& batchMs,
        int totalVecs) {
    AppendStats out;
    if (batchMs.empty()) {
        return out;
    }

    out.numBatches = (int)batchMs.size();
    out.totalMs = std::accumulate(batchMs.begin(), batchMs.end(), 0.0);
    out.avgBatchMs = out.totalMs / out.numBatches;
    out.vecsPerSec = (out.totalMs > 0.0) ? (1000.0 * totalVecs / out.totalMs) : 0.0;

    const int q = std::max(1, out.numBatches / 4);
    double firstSum = 0.0;
    double lastSum = 0.0;
    for (int i = 0; i < q; ++i) {
        firstSum += batchMs[i];
        lastSum += batchMs[out.numBatches - q + i];
    }
    out.firstQuartileMs = firstSum / q;
    out.lastQuartileMs = lastSum / q;
    out.lastOverFirst = (out.firstQuartileMs > 0.0)
            ? (out.lastQuartileMs / out.firstQuartileMs)
            : 0.0;
    return out;
}

template <typename AddFn>
AppendStats runAppendBatches(
        AddFn&& addFn,
        const std::vector<float>& base,
        const std::vector<faiss::idx_t>& ids,
        int d,
        int totalVecs,
        int batchSize) {
    std::vector<double> batchMs;
    batchMs.reserve((size_t)((totalVecs + batchSize - 1) / batchSize));

    int offset = 0;
    while (offset < totalVecs) {
        const int n = std::min(batchSize, totalVecs - offset);
        const float* x = base.data() + (size_t)offset * d;
        const faiss::idx_t* xids = ids.data() + offset;

        const double t0 = nowSec();
        addFn(n, x, xids);
        const double ms = (nowSec() - t0) * 1000.0;
        batchMs.push_back(ms);

        offset += n;
    }

    return summarizeBatchTimes(batchMs, totalVecs);
}

void printStats(
        const char* impl,
        int d,
        int nlist,
        int totalVecs,
        int batchSize,
        const AppendStats& st) {
    printf("%s add_with_ids: total=%d batch=%d batches=%d\n",
           impl, totalVecs, batchSize, st.numBatches);
    printf("  avg batch: %.3f ms | vec/s: %.0f | q1: %.3f ms | q4: %.3f ms | q4/q1: %.3f\n",
           st.avgBatchMs, st.vecsPerSec, st.firstQuartileMs, st.lastQuartileMs,
           st.lastOverFirst);
    printf(
            "SUMMARY,index=IVFFlatAdd,impl=%s,d=%d,nlist=%d,total=%d,batch=%d,batches=%d,"
            "avg_batch_ms=%.6f,vecs_per_sec=%.2f,first_quartile_ms=%.6f,last_quartile_ms=%.6f,"
            "last_over_first=%.6f\n",
            impl, d, nlist, totalVecs, batchSize, st.numBatches, st.avgBatchMs,
            st.vecsPerSec, st.firstQuartileMs, st.lastQuartileMs, st.lastOverFirst);
}

} // namespace

int main(int argc, char** argv) {
    int d = 128;
    int nlist = 1024;
    int totalVecs = 65536;
    int trainVecs = 65536;
    int maxBatches = 256;
    std::vector<int> batchSizes = {128, 512, 2048, 8192};
    std::vector<int> autoReserveSweep = {0, 512, 1024, 2048, 4096};
    int singleAutoReserveMinBatch = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--total") == 0 && i + 1 < argc) {
            totalVecs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            trainVecs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            d = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nlist") == 0 && i + 1 < argc) {
            nlist = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-batches") == 0 && i + 1 < argc) {
            maxBatches = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--auto-reserve-min-batch") == 0 &&
                   i + 1 < argc) {
            singleAutoReserveMinBatch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--auto-reserve-sweep") == 0 &&
                   i + 1 < argc) {
            autoReserveSweep = parseIntList(argv[++i]);
        }
    }
    if (singleAutoReserveMinBatch >= 0) {
        autoReserveSweep = {singleAutoReserveMinBatch};
    }
    if (autoReserveSweep.empty()) {
        autoReserveSweep = {1024};
    }

    printf("=== Faiss Metal vs CPU IVFFlat Append Benchmark ===\n\n");
    printf("Config: d=%d nlist=%d train=%d total_add=%d\n", d, nlist, trainVecs, totalVecs);
    printf("Batch sizes:");
    for (int bs : batchSizes) {
        printf(" %d", bs);
    }
    printf("\nAuto-reserve sweep:");
    for (int t : autoReserveSweep) {
        printf(" %d", t);
    }
    printf("\n\n");

    auto metalRes = std::make_shared<faiss::gpu_metal::StandardMetalResources>();
    if (!metalRes->getResources() || !metalRes->getResources()->isAvailable()) {
        printf("Metal not available. Skipping runs.\n");
        return 1;
    }

    std::vector<float> trainData((size_t)trainVecs * d);
    std::vector<float> addData((size_t)totalVecs * d);
    std::vector<faiss::idx_t> addIds((size_t)totalVecs);
    faiss::float_rand(trainData.data(), trainData.size(), 7001);
    faiss::float_rand(addData.data(), addData.size(), 7002);
    for (int i = 0; i < totalVecs; ++i) {
        addIds[i] = i;
    }

    for (int batchSize : batchSizes) {
        const int numBatches = (totalVecs + batchSize - 1) / batchSize;
        if (numBatches > maxBatches) {
            printf(
                    "=== batch = %d ===\n"
                    "skipping: batches=%d exceeds --max-batches=%d "
                    "(increase batch size or raise --max-batches)\n\n",
                    batchSize,
                    numBatches,
                    maxBatches);
            continue;
        }

        printf("=== batch = %d ===\n", batchSize);

        faiss::IndexFlatL2 coarseCpu(d);
        faiss::IndexIVFFlat cpuIndex(&coarseCpu, d, nlist, faiss::METRIC_L2);
        cpuIndex.train(trainVecs, trainData.data());

        AppendStats cpuStats = runAppendBatches(
                [&](int n, const float* x, const faiss::idx_t* ids) {
                    cpuIndex.add_with_ids(n, x, ids);
                },
                addData,
                addIds,
                d,
                totalVecs,
                batchSize);

        printStats("cpu", d, nlist, totalVecs, batchSize, cpuStats);

        const char* oldThresholdEnv = std::getenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH");
        std::string oldThreshold = oldThresholdEnv ? oldThresholdEnv : "";
        const bool hadOldThreshold = oldThresholdEnv != nullptr;

        for (int threshold : autoReserveSweep) {
            char thresholdBuf[32];
            std::snprintf(thresholdBuf, sizeof(thresholdBuf), "%d", threshold);
            setenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH", thresholdBuf, 1);

            AppendStats metalStats;
            faiss::gpu_metal::MetalIndexIVFFlat::AppendDebugStats dbg;
            @autoreleasepool {
                faiss::gpu_metal::MetalIndexIVFFlat thresholdMetalIndex(
                        metalRes->getResources(), d, nlist, faiss::METRIC_L2, 0.0f);
                thresholdMetalIndex.train(trainVecs, trainData.data());
                thresholdMetalIndex.resetAppendDebugStats();

                metalStats = runAppendBatches(
                        [&](int n, const float* x, const faiss::idx_t* ids) {
                            thresholdMetalIndex.add_with_ids(n, x, ids);
                        },
                        addData,
                        addIds,
                        d,
                        totalVecs,
                        batchSize);
                dbg = thresholdMetalIndex.appendDebugStats();
            }

            printf("auto_reserve_min_batch=%d\n", threshold);
            printStats("metal", d, nlist, totalVecs, batchSize, metalStats);
            const double speedup =
                    (metalStats.avgBatchMs > 0.0)
                    ? (cpuStats.avgBatchMs / metalStats.avgBatchMs)
                    : 0.0;
            printf("speedup (CPU/Metal): %.3fx\n", speedup);
            printf(
                    "allocator: relayouts=%zu moved_lists=%zu moved_vecs=%zu "
                    "reuse_allocs=%zu tail_allocs=%zu reuse_vecs=%zu tail_vecs=%zu "
                    "tail_shrink_events=%zu tail_shrunk_vecs=%zu\n",
                    dbg.relayoutEvents,
                    dbg.movedLists,
                    dbg.movedVectors,
                    dbg.reusedSegmentAllocs,
                    dbg.tailSegmentAllocs,
                    dbg.reusedCapacityVecs,
                    dbg.tailCapacityVecs,
                    dbg.tailShrinkEvents,
                    dbg.tailShrunkVecs);
            printf(
                    "SUMMARY,index=IVFFlatAddCompare,d=%d,nlist=%d,total=%d,batch=%d,"
                    "auto_reserve_min_batch=%d,speedup_cpu_over_metal=%.6f,"
                    "cpu_q4_over_q1=%.6f,metal_q4_over_q1=%.6f,"
                    "relayouts=%zu,moved_lists=%zu,moved_vecs=%zu,"
                    "reuse_allocs=%zu,tail_allocs=%zu,reuse_vecs=%zu,tail_vecs=%zu,"
                    "tail_shrink_events=%zu,tail_shrunk_vecs=%zu\n",
                    d,
                    nlist,
                    totalVecs,
                    batchSize,
                    threshold,
                    speedup,
                    cpuStats.lastOverFirst,
                    metalStats.lastOverFirst,
                    dbg.relayoutEvents,
                    dbg.movedLists,
                    dbg.movedVectors,
                    dbg.reusedSegmentAllocs,
                    dbg.tailSegmentAllocs,
                    dbg.reusedCapacityVecs,
                    dbg.tailCapacityVecs,
                    dbg.tailShrinkEvents,
                    dbg.tailShrunkVecs);
        }

        if (hadOldThreshold) {
            setenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH", oldThreshold.c_str(), 1);
        } else {
            unsetenv("FAISS_METAL_IVF_AUTO_RESERVE_MIN_BATCH");
        }
        printf("\n");
    }

    printf("==================\n");
    return 0;
}
