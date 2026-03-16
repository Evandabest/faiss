// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalResources.h"

#include <cstdlib>

namespace faiss {
namespace gpu_metal {

MetalResources::MetalResources()
        : device_(nil),
          commandQueue_(nil),
          tempPoolBuckets_(nil),
          tempPoolBudgetBytes_(kDefaultTempPoolBudgetBytes),
          tempPoolCachedBytes_(0) {
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

id<MTLBuffer> MetalResources::allocTemporaryBuffer_(size_t size) {
    const size_t wantSize = alignTempBufferSize_(size);
    std::lock_guard<std::mutex> guard(tempPoolMutex_);

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
            return buffer;
        }
    }

    return [device_ newBufferWithLength:wantSize
                                options:MTLResourceStorageModeShared];
}

void MetalResources::deallocTemporaryBuffer_(id<MTLBuffer> buffer) {
    if (buffer == nil) {
        return;
    }
    std::lock_guard<std::mutex> guard(tempPoolMutex_);
    if (tempPoolBudgetBytes_ == 0 || tempPoolBuckets_ == nil) {
        return;
    }

    const size_t size = alignTempBufferSize_((size_t)[buffer length]);
    if (size == 0) {
        return;
    }

    if (tempPoolCachedBytes_ + size > tempPoolBudgetBytes_) {
        return;
    }

    NSNumber* key = @(size);
    NSMutableArray<id<MTLBuffer>>* bucket = tempPoolBuckets_[key];
    if (bucket == nil) {
        bucket = [NSMutableArray array];
        tempPoolBuckets_[key] = bucket;
    }
    [bucket addObject:buffer];
    tempPoolCachedBytes_ += size;
}

id<MTLBuffer> MetalResources::allocBuffer(size_t size, MetalAllocType type) {
    if (!device_) {
        return nil;
    }

    if (type == MetalAllocType::TemporaryMemoryBuffer) {
        return allocTemporaryBuffer_(size);
    }
    if (type == MetalAllocType::TemporaryMemoryOverflow) {
        return [device_ newBufferWithLength:alignTempBufferSize_(size)
                                    options:MTLResourceStorageModeShared];
    }

    return [device_ newBufferWithLength:size
                                options:MTLResourceStorageModeShared];
}

void MetalResources::deallocBuffer(id<MTLBuffer> buffer, MetalAllocType type) {
    if (type == MetalAllocType::TemporaryMemoryBuffer) {
        deallocTemporaryBuffer_(buffer);
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
