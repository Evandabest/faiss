// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MetalKernels implementation: library loading, pipeline caching, and typed
 * encode methods for every Metal kernel used by the Faiss Metal backend.
 */

#import "MetalKernels.h"
#include <algorithm>
#include <mutex>

namespace faiss {
namespace gpu_metal {

namespace {

static NSString* loadMSLSource() {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* metalPath = nil;

    NSBundle* mainBundle = [NSBundle mainBundle];
    metalPath = [mainBundle pathForResource:@"MetalDistance" ofType:@"metal"];

    if (!metalPath) {
        NSString* cwd = [fm currentDirectoryPath];
        NSString* relPath = [cwd stringByAppendingPathComponent:
                @"faiss/gpu_metal/MetalDistance.metal"];
        if ([fm fileExistsAtPath:relPath]) {
            metalPath = relPath;
        }
    }

    if (!metalPath) {
        NSString* sourceFile = @(__FILE__);
        NSString* sourceDir = [sourceFile stringByDeletingLastPathComponent];
        NSString* relPath = [sourceDir stringByAppendingPathComponent:
                @"MetalDistance.metal"];
        if ([fm fileExistsAtPath:relPath]) {
            metalPath = relPath;
        }
    }

    if (!metalPath) {
        NSString* execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* relPath = [execDir stringByAppendingPathComponent:
                    @"MetalDistance.metal"];
            if ([fm fileExistsAtPath:relPath]) {
                metalPath = relPath;
            }
        }
    }

    if (metalPath) {
        NSError* err = nil;
        NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&err];
        if (source && !err) {
            return source;
        }
    }

    return nil;
}

static const char* kThreadgroupNames[] = {
        "topk_threadgroup_32",  "topk_threadgroup_64",
        "topk_threadgroup_128", "topk_threadgroup_256",
        "topk_threadgroup_512", "topk_threadgroup_1024",
        "topk_threadgroup_2048"};

static const char* kFusedDistTopKNames[] = {
        "fused_dist_topk_32",  "fused_dist_topk_64",
        "fused_dist_topk_128", "fused_dist_topk_256",
        "fused_dist_topk_512", "fused_dist_topk_1024",
        nullptr};

static const char* kFusedDistTopKFP16Names[] = {
        "fused_dist_topk_fp16_32",  "fused_dist_topk_fp16_64",
        "fused_dist_topk_fp16_128", "fused_dist_topk_fp16_256",
        "fused_dist_topk_fp16_512", "fused_dist_topk_fp16_1024",
        nullptr};

static const char* kBitonicMergeNames[] = {
        "topk_merge_two_sorted_32",  "topk_merge_two_sorted_64",
        "topk_merge_two_sorted_128", "topk_merge_two_sorted_256",
        "topk_merge_two_sorted_512", "topk_merge_two_sorted_1024",
        nullptr};

} // namespace

// ============================================================
//  Statics
// ============================================================

constexpr int MetalKernels::kTopKVariantSizes[];

int MetalKernels::selectTopKVariantIndex(int k) {
    for (int i = 0; i < kNumTopKVariants; i++) {
        if (k <= kTopKVariantSizes[i])
            return i;
    }
    return kNumTopKVariants - 1;
}

int MetalKernels::computeKPrimeForTiling(int k) {
    int cap = std::min(2 * k, 1024);
    if (cap <= 0) return 32;
    uint32_t x = (uint32_t)cap;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x++;
    return (int)std::min(std::max(x, 32u), 1024u);
}

// ============================================================
//  Constructor / destructor
// ============================================================

MetalKernels::MetalKernels(id<MTLDevice> device)
        : device_(device), library_(nil) {
    NSString* src = loadMSLSource();
    if (!src) return;
    NSError* err = nil;
    library_ = [device_ newLibraryWithSource:src options:nil error:&err];
}

MetalKernels::~MetalKernels() = default;

bool MetalKernels::isValid() const {
    return library_ != nil;
}

id<MTLComputePipelineState> MetalKernels::pipeline(const char* name) {
    std::string key(name);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    id<MTLFunction> fn = [library_ newFunctionWithName:@(name)];
    if (!fn) return nil;
    NSError* err = nil;
    id<MTLComputePipelineState> ps =
            [device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) return nil;
    cache_[key] = ps;
    return ps;
}

// ============================================================
//  Distance matrix ops
// ============================================================

void MetalKernels::encodeL2SquaredMatrix(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        int nq, int nb, int d,
        size_t queryByteOff, size_t vectorByteOff) {
    [enc setComputePipelineState:pipeline("l2_squared_matrix")];
    [enc setBuffer:queries   offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors   offset:vectorByteOff atIndex:1];
    [enc setBuffer:distances offset:0              atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    const NSUInteger tileM = 32, tileN = 32;
    MTLSize grid = MTLSizeMake(((NSUInteger)nb + tileN - 1) / tileN,
                                ((NSUInteger)nq + tileM - 1) / tileM, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void MetalKernels::encodeIPMatrix(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        int nq, int nb, int d,
        size_t queryByteOff, size_t vectorByteOff) {
    [enc setComputePipelineState:pipeline("ip_matrix")];
    [enc setBuffer:queries   offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors   offset:vectorByteOff atIndex:1];
    [enc setBuffer:distances offset:0              atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    const NSUInteger tileM = 32, tileN = 32;
    MTLSize grid = MTLSizeMake(((NSUInteger)nb + tileN - 1) / tileN,
                                ((NSUInteger)nq + tileM - 1) / tileM, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void MetalKernels::encodeL2WithNorms(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        id<MTLBuffer> vecNorms,
        int nq, int nb, int d) {
    [enc setComputePipelineState:pipeline("l2_with_norms")];
    [enc setBuffer:queries   offset:0 atIndex:0];
    [enc setBuffer:vectors   offset:0 atIndex:1];
    [enc setBuffer:distances offset:0 atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    [enc setBuffer:vecNorms offset:0 atIndex:4];
    const NSUInteger w = 16, h = 16;
    MTLSize grid = MTLSizeMake(((NSUInteger)nb + w - 1) / w,
                                ((NSUInteger)nq + h - 1) / h, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}

void MetalKernels::encodeComputeNorms(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> vectors,
        id<MTLBuffer> norms,
        int nb, int d) {
    auto ps = pipeline("compute_norms");
    [enc setComputePipelineState:ps];
    [enc setBuffer:vectors offset:0 atIndex:0];
    [enc setBuffer:norms   offset:0 atIndex:1];
    uint32_t args[2] = {(uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:2];
    NSUInteger tgSize = std::min((NSUInteger)256,
                                  ps.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)nb + tgSize - 1) / tgSize;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
}

// ============================================================
//  Fused distance + top-k
// ============================================================

void MetalKernels::encodeFusedDistTopK(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        int nq, int nb, int d, int k, bool isL2,
        size_t queryByteOff, size_t vectorByteOff) {
    int vi = selectTopKVariantIndex(k);
    const char* name = kFusedDistTopKNames[vi];
    if (!name) return;
    [enc setComputePipelineState:pipeline(name)];
    [enc setBuffer:queries offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors offset:vectorByteOff atIndex:1];
    [enc setBuffer:outDist offset:0             atIndex:2];
    [enc setBuffer:outIdx  offset:0             atIndex:3];
    uint32_t args[5] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d,
                         (uint32_t)k, isL2 ? 0u : 1u};
    [enc setBytes:args length:sizeof(args) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================
//  Float16 vector distance + fused top-k
// ============================================================

void MetalKernels::encodeL2SquaredMatrixFP16(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        int nq, int nb, int d,
        size_t queryByteOff, size_t vectorByteOff) {
    [enc setComputePipelineState:pipeline("l2_squared_matrix_fp16")];
    [enc setBuffer:queries   offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors   offset:vectorByteOff atIndex:1];
    [enc setBuffer:distances offset:0              atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    const NSUInteger tileM = 32, tileN = 32;
    MTLSize grid = MTLSizeMake(((NSUInteger)nb + tileN - 1) / tileN,
                                ((NSUInteger)nq + tileM - 1) / tileM, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void MetalKernels::encodeIPMatrixFP16(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> distances,
        int nq, int nb, int d,
        size_t queryByteOff, size_t vectorByteOff) {
    [enc setComputePipelineState:pipeline("ip_matrix_fp16")];
    [enc setBuffer:queries   offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors   offset:vectorByteOff atIndex:1];
    [enc setBuffer:distances offset:0              atIndex:2];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    const NSUInteger tileM = 32, tileN = 32;
    MTLSize grid = MTLSizeMake(((NSUInteger)nb + tileN - 1) / tileN,
                                ((NSUInteger)nq + tileM - 1) / tileM, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void MetalKernels::encodeFusedDistTopKFP16(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        int nq, int nb, int d, int k, bool isL2,
        size_t queryByteOff, size_t vectorByteOff) {
    int vi = selectTopKVariantIndex(k);
    const char* name = kFusedDistTopKFP16Names[vi];
    if (!name) return;
    [enc setComputePipelineState:pipeline(name)];
    [enc setBuffer:queries offset:queryByteOff  atIndex:0];
    [enc setBuffer:vectors offset:vectorByteOff atIndex:1];
    [enc setBuffer:outDist offset:0             atIndex:2];
    [enc setBuffer:outIdx  offset:0             atIndex:3];
    uint32_t args[5] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)d,
                         (uint32_t)k, isL2 ? 0u : 1u};
    [enc setBytes:args length:sizeof(args) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================
//  Top-k ops
// ============================================================

void MetalKernels::encodeTopKThreadgroup(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> distances,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        int nq, int nb, int k, bool wantMin) {
    int vi = selectTopKVariantIndex(k);
    [enc setComputePipelineState:pipeline(kThreadgroupNames[vi])];
    [enc setBuffer:distances offset:0 atIndex:0];
    [enc setBuffer:outDist   offset:0 atIndex:1];
    [enc setBuffer:outIdx    offset:0 atIndex:2];
    uint32_t args[4] = {(uint32_t)nq, (uint32_t)nb, (uint32_t)k,
                         wantMin ? 1u : 0u};
    [enc setBytes:args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================
//  Merge ops
// ============================================================

void MetalKernels::encodeMergeTwoSorted(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> inA, id<MTLBuffer> inAIdx,
        id<MTLBuffer> inB, id<MTLBuffer> inBIdx,
        id<MTLBuffer> out, id<MTLBuffer> outIdx,
        int nq, int kActual, bool wantMin,
        size_t inAOff, size_t inAIdxOff,
        size_t inBOff, size_t inBIdxOff,
        size_t outOff, size_t outIdxOff) {
    int vi = selectTopKVariantIndex(kActual);
    const char* name = kBitonicMergeNames[vi];
    if (!name) return;
    [enc setComputePipelineState:pipeline(name)];
    [enc setBuffer:inA    offset:inAOff    atIndex:0];
    [enc setBuffer:inAIdx offset:inAIdxOff atIndex:1];
    [enc setBuffer:inB    offset:inBOff    atIndex:2];
    [enc setBuffer:inBIdx offset:inBIdxOff atIndex:3];
    [enc setBuffer:out    offset:outOff    atIndex:4];
    [enc setBuffer:outIdx offset:outIdxOff atIndex:5];
    uint32_t args[3] = {(uint32_t)nq, (uint32_t)kActual, wantMin ? 1u : 0u};
    [enc setBytes:args length:sizeof(args) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalKernels::encodeIncrementIndex(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> indices,
        int nq, int k, int tileCols, int numTiles,
        size_t indicesOff) {
    [enc setComputePipelineState:pipeline("increment_index")];
    [enc setBuffer:indices offset:indicesOff atIndex:0];
    uint32_t args[4] = {(uint32_t)nq, (uint32_t)k, (uint32_t)tileCols,
                         (uint32_t)numTiles};
    [enc setBytes:args length:sizeof(args) atIndex:1];
    MTLSize grid = MTLSizeMake((NSUInteger)numTiles, (NSUInteger)nq, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void MetalKernels::encodeTrimKToK(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> inK, id<MTLBuffer> inIdx,
        id<MTLBuffer> outK, id<MTLBuffer> outIdx,
        int nq, int kPrime, int k, bool wantMin,
        size_t inKOff, size_t inIdxOff,
        size_t outKOff, size_t outIdxOff) {
    [enc setComputePipelineState:pipeline("trim_K_to_k")];
    [enc setBuffer:inK    offset:inKOff    atIndex:0];
    [enc setBuffer:inIdx  offset:inIdxOff  atIndex:1];
    [enc setBuffer:outK   offset:outKOff   atIndex:2];
    [enc setBuffer:outIdx offset:outIdxOff atIndex:3];
    uint32_t args[4] = {(uint32_t)nq, (uint32_t)kPrime, (uint32_t)k,
                         wantMin ? 1u : 0u};
    [enc setBytes:args length:sizeof(args) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

// ============================================================
//  IVF ops
// ============================================================

void MetalKernels::encodeIVFScanList(
        id<MTLComputeCommandEncoder> enc,
        IVFScanVariant variant,
        id<MTLBuffer> queries, id<MTLBuffer> codes, id<MTLBuffer> ids,
        id<MTLBuffer> listOffset, id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> perListDist, id<MTLBuffer> perListIdx,
        id<MTLBuffer> paramsBuf,
        int nq, int nprobe,
        id<MTLBuffer> ilCodesOffset,
        id<MTLBuffer> sqTables) {
    const char* name;
    NSUInteger tgSize;
    switch (variant) {
        case IVFScanVariant::Small:
            name = "ivf_scan_list_small"; tgSize = 32; break;
        case IVFScanVariant::Interleaved:
            name = "ivf_scan_list_interleaved"; tgSize = 256; break;
        case IVFScanVariant::SQ4:
            name = "ivf_scan_list_sq4"; tgSize = 256; break;
        case IVFScanVariant::SQ6:
            name = "ivf_scan_list_sq6"; tgSize = 256; break;
        case IVFScanVariant::SQ8:
            name = "ivf_scan_list_sq8"; tgSize = 256; break;
        case IVFScanVariant::FP16:
            name = "ivf_scan_list_fp16"; tgSize = 256; break;
        default:
            name = "ivf_scan_list"; tgSize = 256; break;
    }
    [enc setComputePipelineState:pipeline(name)];
    [enc setBuffer:queries      offset:0 atIndex:0];
    [enc setBuffer:codes        offset:0 atIndex:1];
    [enc setBuffer:ids          offset:0 atIndex:2];
    [enc setBuffer:listOffset   offset:0 atIndex:3];
    [enc setBuffer:listLength   offset:0 atIndex:4];
    [enc setBuffer:coarseAssign offset:0 atIndex:5];
    [enc setBuffer:perListDist  offset:0 atIndex:6];
    [enc setBuffer:perListIdx   offset:0 atIndex:7];
    [enc setBuffer:paramsBuf    offset:0 atIndex:8];
    if (variant == IVFScanVariant::Interleaved && ilCodesOffset) {
        [enc setBuffer:ilCodesOffset offset:0 atIndex:9];
    }
    if ((variant == IVFScanVariant::SQ4 ||
         variant == IVFScanVariant::SQ6 ||
         variant == IVFScanVariant::SQ8) && sqTables) {
        [enc setBuffer:sqTables offset:0 atIndex:9];
    }
    NSUInteger totalTGs = (NSUInteger)nq * (NSUInteger)nprobe;
    [enc dispatchThreadgroups:MTLSizeMake(totalTGs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
}

void MetalKernels::encodeIVFPQScanList(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> lookupTable,
        id<MTLBuffer> codes,
        id<MTLBuffer> ids,
        id<MTLBuffer> listOffset,
        id<MTLBuffer> listLength,
        id<MTLBuffer> coarseAssign,
        id<MTLBuffer> perListDist,
        id<MTLBuffer> perListIdx,
        id<MTLBuffer> paramsBuf,
        int nq,
        int nprobe) {
    [enc setComputePipelineState:pipeline("ivf_scan_list_pq8")];
    [enc setBuffer:lookupTable  offset:0 atIndex:0];
    [enc setBuffer:codes        offset:0 atIndex:1];
    [enc setBuffer:ids          offset:0 atIndex:2];
    [enc setBuffer:listOffset   offset:0 atIndex:3];
    [enc setBuffer:listLength   offset:0 atIndex:4];
    [enc setBuffer:coarseAssign offset:0 atIndex:5];
    [enc setBuffer:perListDist  offset:0 atIndex:6];
    [enc setBuffer:perListIdx   offset:0 atIndex:7];
    [enc setBuffer:paramsBuf    offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq * (NSUInteger)nprobe, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalKernels::encodeHammingDistanceTopK(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> queries,
        id<MTLBuffer> database,
        id<MTLBuffer> outDist,
        id<MTLBuffer> outIdx,
        id<MTLBuffer> paramsBuf,
        int nq) {
    [enc setComputePipelineState:pipeline("hamming_distance_topk")];
    [enc setBuffer:queries   offset:0 atIndex:0];
    [enc setBuffer:database  offset:0 atIndex:1];
    [enc setBuffer:outDist   offset:0 atIndex:2];
    [enc setBuffer:outIdx    offset:0 atIndex:3];
    [enc setBuffer:paramsBuf offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalKernels::encodeIVFMergeLists(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> perListDist, id<MTLBuffer> perListIdx,
        id<MTLBuffer> outDist, id<MTLBuffer> outIdx,
        id<MTLBuffer> paramsBuf,
        int nq) {
    [enc setComputePipelineState:pipeline("ivf_merge_lists")];
    [enc setBuffer:perListDist offset:0 atIndex:0];
    [enc setBuffer:perListIdx  offset:0 atIndex:1];
    [enc setBuffer:outDist     offset:0 atIndex:2];
    [enc setBuffer:outIdx      offset:0 atIndex:3];
    [enc setBuffer:paramsBuf   offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================
//  Per-device singleton
// ============================================================

MetalKernels& getMetalKernels(id<MTLDevice> device) {
    static std::mutex mu;
    static std::unordered_map<uintptr_t, std::unique_ptr<MetalKernels>> map;
    uintptr_t key = (uintptr_t)(__bridge void*)device;
    std::lock_guard<std::mutex> lock(mu);
    auto& ptr = map[key];
    if (!ptr) ptr = std::make_unique<MetalKernels>(device);
    return *ptr;
}

} // namespace gpu_metal
} // namespace faiss
