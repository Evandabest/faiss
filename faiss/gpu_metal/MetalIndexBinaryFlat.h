// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal binary flat index: brute-force Hamming distance search on GPU.
 */

#pragma once

#import <Metal/Metal.h>

#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu_metal/MetalResources.h>

#include <memory>
#include <vector>

namespace faiss {
namespace gpu_metal {

class MetalIndexBinaryFlat : public faiss::IndexBinary {
public:
    MetalIndexBinaryFlat(
            std::shared_ptr<MetalResources> resources,
            int d);

    MetalIndexBinaryFlat(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexBinaryFlat* cpuIndex);

    ~MetalIndexBinaryFlat() override;

    void add(idx_t n, const uint8_t* x) override;
    void reset() override;

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, uint8_t* recons) const override;

    void copyFrom(const faiss::IndexBinaryFlat* index);
    void copyTo(faiss::IndexBinaryFlat* index) const;

    std::shared_ptr<MetalResources> getResources() { return resources_; }

private:
    std::shared_ptr<MetalResources> resources_;

    std::vector<uint8_t> hostVectors_;

    id<MTLBuffer> vectorsBuf_ = nil;
    mutable id<MTLBuffer> queryBuf_   = nil;
    mutable id<MTLBuffer> outDistBuf_ = nil;
    mutable id<MTLBuffer> outIdxBuf_  = nil;
    mutable size_t queryBufCap_   = 0;
    mutable size_t outDistBufCap_ = 0;
    mutable size_t outIdxBufCap_  = 0;

    void uploadToGpu();
    void ensureBuf_(id<MTLBuffer>& buf, size_t& cap, size_t needed) const;
};

} // namespace gpu_metal
} // namespace faiss
