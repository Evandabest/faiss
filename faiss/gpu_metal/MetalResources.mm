// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalResources.h"

#include <cstdio>
#include <cstdlib>

namespace faiss {
namespace gpu_metal {

MetalResources::MetalResources()
        : device_(nil),
          commandQueue_(nil),
          tempPoolBuckets_(nil),
          tempPoolBudgetBytes_(kDefaultTempPoolBudgetBytes),
          tempPoolCachedBytes_(0),
          allocLogging_(false) {
    const char* envBytes = std::getenv("FAISS_METAL_TEMP_POOL_BYTES");
    if (envBytes && envBytes[0] != '\0') {
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(envBytes, &end, 10);
        if (end != envBytes) {
            tempPoolBudgetBytes_ = static_cast<size_t>(parsed);
        }
    }

    device_ = MTLCreateSystemDefaultDevice();
    if (device_) {
        commandQueue_ = [device_ newCommandQueue];
        tempPoolBuckets_ = [NSMutableDictionary dictionary];
    }
}

MetalResources::~MetalResources() {
    clearTempMemoryPool();
    tempPoolBuckets_ = nil;
    commandQueue_ = nil;
    device_ = nil;
}

size_t MetalResources::alignTempBufferSize_(size_t bytes) const {
    if (bytes == 0) {
        return 0;
    }
    const size_t a = kTempPoolAlignBytes;
    return ((bytes + a - 1) / a) * a;
}

id<MTLBuffer> MetalResources::allocTemporaryBuffer_(
        size_t size,
        bool* reusedFromPool) {
    const size_t wantSize = alignTempBufferSize_(size);
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    if (reusedFromPool) {
        *reusedFromPool = false;
    }

    if (tempPoolBudgetBytes_ > 0 && tempPoolBuckets_ != nil) {
        NSNumber* bestKey = nil;
        for (NSNumber* key in tempPoolBuckets_) {
            const size_t keySize = (size_t)key.unsignedLongLongValue;
            if (keySize < wantSize) {
                continue;
            }
            NSMutableArray<id<MTLBuffer>>* bucket = tempPoolBuckets_[key];
            if (bucket.count == 0) {
                continue;
            }
            if (bestKey == nil ||
                keySize < (size_t)bestKey.unsignedLongLongValue) {
                bestKey = key;
            }
        }

        if (bestKey != nil) {
            NSMutableArray<id<MTLBuffer>>* bucket = tempPoolBuckets_[bestKey];
            id<MTLBuffer> buffer = [bucket lastObject];
            [bucket removeLastObject];

            const size_t bufSize = (size_t)bestKey.unsignedLongLongValue;
            if (tempPoolCachedBytes_ >= bufSize) {
                tempPoolCachedBytes_ -= bufSize;
            } else {
                tempPoolCachedBytes_ = 0;
            }
            if (bucket.count == 0) {
                [tempPoolBuckets_ removeObjectForKey:bestKey];
            }
            if (reusedFromPool) {
                *reusedFromPool = true;
            }
            return buffer;
        }
    }

    return [device_ newBufferWithLength:wantSize
                                options:MTLResourceStorageModeShared];
}

bool MetalResources::deallocTemporaryBuffer_(id<MTLBuffer> buffer) {
    if (buffer == nil) {
        return false;
    }
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    if (tempPoolBudgetBytes_ == 0 || tempPoolBuckets_ == nil) {
        return false;
    }

    const size_t size = alignTempBufferSize_((size_t)[buffer length]);
    if (size == 0) {
        return false;
    }

    if (tempPoolCachedBytes_ + size > tempPoolBudgetBytes_) {
        return false;
    }

    NSNumber* key = @(size);
    NSMutableArray<id<MTLBuffer>>* bucket = tempPoolBuckets_[key];
    if (bucket == nil) {
        bucket = [NSMutableArray array];
        tempPoolBuckets_[key] = bucket;
    }
    [bucket addObject:buffer];
    tempPoolCachedBytes_ += size;
    return true;
}

id<MTLBuffer> MetalResources::allocBuffer(size_t size, MetalAllocType type) {
    if (!device_) {
        return nil;
    }

    if (type == MetalAllocType::TemporaryMemoryBuffer) {
        bool reused = false;
        id<MTLBuffer> buffer = allocTemporaryBuffer_(size, &reused);
        recordAlloc_(buffer, type, alignTempBufferSize_(size));
        if (buffer && getLogMemoryAllocations()) {
            std::fprintf(
                    stderr,
                    "[faiss_metal] alloc type=%s req=%zu alloc=%zu src=%s ptr=%p\n",
                    allocTypeName_(type),
                    size,
                    (size_t)[buffer length],
                    reused ? "pool" : "fresh",
                    [buffer contents]);
        }
        return buffer;
    }
    if (type == MetalAllocType::TemporaryMemoryOverflow) {
        id<MTLBuffer> buffer =
                [device_ newBufferWithLength:alignTempBufferSize_(size)
                                     options:MTLResourceStorageModeShared];
        recordAlloc_(buffer, type, alignTempBufferSize_(size));
        if (buffer && getLogMemoryAllocations()) {
            std::fprintf(
                    stderr,
                    "[faiss_metal] alloc type=%s req=%zu alloc=%zu src=overflow ptr=%p\n",
                    allocTypeName_(type),
                    size,
                    (size_t)[buffer length],
                    [buffer contents]);
        }
        return buffer;
    }

    id<MTLBuffer> buffer =
            [device_ newBufferWithLength:size
                                 options:MTLResourceStorageModeShared];
    recordAlloc_(buffer, type, size);
    if (buffer && getLogMemoryAllocations()) {
        std::fprintf(
                stderr,
                "[faiss_metal] alloc type=%s req=%zu alloc=%zu src=direct ptr=%p\n",
                allocTypeName_(type),
                size,
                (size_t)[buffer length],
                [buffer contents]);
    }
    return buffer;
}

void MetalResources::deallocBuffer(id<MTLBuffer> buffer, MetalAllocType type) {
    bool cachedToPool = false;
    if (type == MetalAllocType::TemporaryMemoryBuffer) {
        cachedToPool = deallocTemporaryBuffer_(buffer);
    }
    recordFree_(buffer, type);
    if (buffer && getLogMemoryAllocations()) {
        std::fprintf(
                stderr,
                "[faiss_metal] free  type=%s alloc=%zu dst=%s ptr=%p\n",
                allocTypeName_(type),
                (size_t)[buffer length],
                cachedToPool ? "pool" : "release",
                [buffer contents]);
    }
}

void MetalResources::setTempMemoryPoolBytes(size_t bytes) {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    tempPoolBudgetBytes_ = bytes;
    if (tempPoolBudgetBytes_ == 0) {
        if (tempPoolBuckets_ != nil) {
            [tempPoolBuckets_ removeAllObjects];
        }
        tempPoolCachedBytes_ = 0;
        return;
    }

    while (tempPoolCachedBytes_ > tempPoolBudgetBytes_) {
        bool removed = false;
        for (NSNumber* key in [tempPoolBuckets_ allKeys]) {
            NSMutableArray<id<MTLBuffer>>* bucket = tempPoolBuckets_[key];
            if (bucket.count == 0) {
                continue;
            }
            [bucket removeLastObject];
            const size_t keySize = (size_t)key.unsignedLongLongValue;
            if (tempPoolCachedBytes_ >= keySize) {
                tempPoolCachedBytes_ -= keySize;
            } else {
                tempPoolCachedBytes_ = 0;
            }
            if (bucket.count == 0) {
                [tempPoolBuckets_ removeObjectForKey:key];
            }
            removed = true;
            break;
        }
        if (!removed) {
            tempPoolCachedBytes_ = 0;
            break;
        }
    }
}

void MetalResources::clearTempMemoryPool() {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    if (tempPoolBuckets_ != nil) {
        [tempPoolBuckets_ removeAllObjects];
    }
    tempPoolCachedBytes_ = 0;
}

size_t MetalResources::getTempMemoryPoolBytes() const {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    return tempPoolBudgetBytes_;
}

size_t MetalResources::getTempMemoryCachedBytes() const {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    return tempPoolCachedBytes_;
}

void MetalResources::setLogMemoryAllocations(bool enable) {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    allocLogging_ = enable;
}

bool MetalResources::getLogMemoryAllocations() const {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    return allocLogging_;
}

size_t MetalResources::trackedAllocTypeIndex_(MetalAllocType type) const {
    const size_t idx = static_cast<size_t>(type);
    if (idx >= kNumTrackedAllocTypes) {
        return static_cast<size_t>(MetalAllocType::Other);
    }
    return idx;
}

const char* MetalResources::allocTypeName_(MetalAllocType type) const {
    switch (type) {
        case MetalAllocType::Other:
            return "Other";
        case MetalAllocType::FlatData:
            return "FlatData";
        case MetalAllocType::IVFLists:
            return "IVFLists";
        case MetalAllocType::Quantizer:
            return "Quantizer";
        case MetalAllocType::QuantizerPrecomputedCodes:
            return "QuantizerPrecomputedCodes";
        case MetalAllocType::TemporaryMemoryBuffer:
            return "TemporaryMemoryBuffer";
        case MetalAllocType::TemporaryMemoryOverflow:
            return "TemporaryMemoryOverflow";
        default:
            return "Unknown";
    }
}

void MetalResources::recordAlloc_(
        id<MTLBuffer> buffer,
        MetalAllocType type,
        size_t bytes) {
    if (buffer == nil) {
        return;
    }
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    const size_t idx = trackedAllocTypeIndex_(type);
    auto& stats = allocStats_[idx];
    stats.liveAllocs++;
    stats.liveBytes += bytes;
    stats.totalAllocs++;
    stats.totalAllocBytes += bytes;
    liveAllocs_[(void*)[buffer contents]] = std::make_pair(type, bytes);
}

void MetalResources::recordFree_(id<MTLBuffer> buffer, MetalAllocType type) {
    if (buffer == nil) {
        return;
    }
    const void* ptr = [buffer contents];
    const size_t fallbackBytes = (size_t)[buffer length];
    std::lock_guard<std::mutex> guard(tempPoolMutex_);

    MetalAllocType trackedType = type;
    size_t bytes = fallbackBytes;
    auto it = liveAllocs_.find((void*)ptr);
    if (it != liveAllocs_.end()) {
        trackedType = it->second.first;
        bytes = it->second.second;
        liveAllocs_.erase(it);
    }

    const size_t idx = trackedAllocTypeIndex_(trackedType);
    auto& stats = allocStats_[idx];
    if (stats.liveAllocs > 0) {
        stats.liveAllocs--;
    }
    if (stats.liveBytes >= bytes) {
        stats.liveBytes -= bytes;
    } else {
        stats.liveBytes = 0;
    }
    stats.totalFrees++;
    stats.totalFreedBytes += bytes;
}

MetalMemoryInfo MetalResources::getMemoryInfo() const {
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    MetalMemoryInfo info;
    info.tempPoolBudgetBytes = tempPoolBudgetBytes_;
    info.tempPoolCachedBytes = tempPoolCachedBytes_;
    info.logMemoryAllocations = allocLogging_;

    for (size_t i = 0; i < allocStats_.size(); ++i) {
        const MetalAllocStats& s = allocStats_[i];
        if (s.liveAllocs == 0 && s.liveBytes == 0 && s.totalAllocs == 0 &&
            s.totalAllocBytes == 0 && s.totalFrees == 0 &&
            s.totalFreedBytes == 0) {
            continue;
        }
        info.byAllocType[(int)i] = s;
        info.totalLiveAllocs += s.liveAllocs;
        info.totalLiveBytes += s.liveBytes;
    }

    return info;
}

void MetalResources::synchronize() {
    if (!commandQueue_) {
        return;
    }
    id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace gpu_metal
} // namespace faiss
