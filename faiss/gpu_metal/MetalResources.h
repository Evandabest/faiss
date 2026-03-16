// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * This header uses Objective-C types (Metal framework: id, nil, MTLDevice, etc.).
 * For correct IDE/linter behavior, associate this file with "Objective-C++":
 *
 */

#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <array>
#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>

namespace faiss {
namespace gpu_metal {

/// Allocation type for Metal buffers (mirrors faiss::gpu::AllocType roles).
enum MetalAllocType {
    Other = 0,
    FlatData = 1,
    IVFLists = 2,
    Quantizer = 3,
    QuantizerPrecomputedCodes = 4,
    TemporaryMemoryBuffer = 10,
    TemporaryMemoryOverflow = 11,
};

struct MetalAllocStats {
    size_t liveAllocs = 0;
    size_t liveBytes = 0;
    size_t totalAllocs = 0;
    size_t totalAllocBytes = 0;
    size_t totalFrees = 0;
    size_t totalFreedBytes = 0;
};

struct MetalMemoryInfo {
    size_t tempPoolBudgetBytes = 0;
    size_t tempPoolCachedBytes = 0;
    bool logMemoryAllocations = false;
    size_t totalLiveAllocs = 0;
    size_t totalLiveBytes = 0;
    std::map<int, MetalAllocStats> byAllocType;
};

/// Owns Metal device, command queue, and provides buffer allocation.
/// Mirrors the roles of faiss::gpu::GpuResources for the Metal backend.
class MetalResources {
public:
    MetalResources();
    ~MetalResources();

    MetalResources(const MetalResources&) = delete;
    MetalResources& operator=(const MetalResources&) = delete;

    /// Returns the Metal device (nil if no Metal-capable device is available).
    id<MTLDevice> getDevice() const { return device_; }

    /// Returns the command queue for the device (nil if device is nil).
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue_; }

    /// Allocates a buffer of the given size (bytes). Caller owns the returned
    /// buffer and must call deallocBuffer when done, or the buffer will leak.
    /// Returns nil on failure (e.g. device nil or allocation failure).
    id<MTLBuffer> allocBuffer(size_t size, MetalAllocType type);

    /// Releases a buffer previously returned by allocBuffer. The caller must
    /// not use the buffer after this call.
    void deallocBuffer(id<MTLBuffer> buffer, MetalAllocType type);

    /// Blocks until all work submitted to the default command queue has completed.
    void synchronize();

    /// Returns true if the Metal device and queue are available.
    bool isAvailable() const { return device_ != nil && commandQueue_ != nil; }

    /// Sets the byte budget used to cache temporary buffers for reuse.
    /// A value of 0 disables temporary-buffer caching.
    void setTempMemoryPoolBytes(size_t bytes);

    /// Returns the temporary buffer cache budget in bytes.
    size_t getTempMemoryPoolBytes() const;

    /// Returns the total bytes currently cached in the temporary pool.
    size_t getTempMemoryCachedBytes() const;

    /// Releases all currently cached temporary buffers.
    void clearTempMemoryPool();

    /// Enables/disables allocation logging to stderr.
    void setLogMemoryAllocations(bool enable);

    /// Returns whether allocation logging is enabled.
    bool getLogMemoryAllocations() const;

    /// Returns a memory snapshot with per-allocation-type stats.
    MetalMemoryInfo getMemoryInfo() const;

private:
    static constexpr size_t kNumTrackedAllocTypes = 12;
    static constexpr size_t kTempPoolAlignBytes = 256;
    static constexpr size_t kDefaultTempPoolBudgetBytes =
            512ULL * 1024 * 1024;

    size_t alignTempBufferSize_(size_t bytes) const;
    id<MTLBuffer> allocTemporaryBuffer_(size_t size, bool* reusedFromPool);
    bool deallocTemporaryBuffer_(id<MTLBuffer> buffer);
    size_t trackedAllocTypeIndex_(MetalAllocType type) const;
    const char* allocTypeName_(MetalAllocType type) const;
    void recordAlloc_(id<MTLBuffer> buffer, MetalAllocType type, size_t bytes);
    void recordFree_(id<MTLBuffer> buffer, MetalAllocType type);

    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    NSMutableDictionary<NSNumber*, NSMutableArray<id<MTLBuffer>>*>*
            tempPoolBuckets_;
    size_t tempPoolBudgetBytes_;
    size_t tempPoolCachedBytes_;
    bool allocLogging_;
    std::unordered_map<void*, std::pair<MetalAllocType, size_t>> liveAllocs_;
    std::array<MetalAllocStats, kNumTrackedAllocTypes> allocStats_;
    mutable std::mutex tempPoolMutex_;
};

} // namespace gpu_metal
} // namespace faiss
