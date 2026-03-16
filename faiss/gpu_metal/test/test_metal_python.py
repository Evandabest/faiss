# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Minimal Python test for the Metal backend. Run when faiss is built with
# FAISS_ENABLE_METAL=ON and FAISS_ENABLE_PYTHON=ON (e.g. from build dir:
#   PYTHONPATH=../.. python test_metal_python.py
# or after installing the Metal-built wheel).

import numpy as np
import unittest
import faiss


class TestMetalPython(unittest.TestCase):
    """Test that the same Python API used for GPU works with the Metal build."""

    def test_get_num_gpus(self):
        n = faiss.get_num_gpus()
        # Metal build: 1 if device available, else 0
        self.assertGreaterEqual(n, 0)
        if n == 0:
            self.skipTest("No Metal device (get_num_gpus() == 0)")

    def test_standard_gpu_resources(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        res = faiss.StandardGpuResources()
        self.assertIsNotNone(res)

    def test_index_cpu_to_gpu_flat_search(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 32, 200, 10, 5
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.assertIsNotNone(gpu_index)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_array_almost_equal(D_gpu, D_cpu)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_index_cpu_to_gpu_with_options(self):
        """Same as above but pass GpuClonerOptions (ignored on Metal) for API parity."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 16, 50, 5, 3
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(xb)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        self.assertIsNotNone(gpu_index)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_array_almost_equal(D_gpu, D_cpu)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_tiled_many_vectors(self):
        """Force vector tiling (nb > 131072): compare Metal vs CPU."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 132_000, 20, 10  # nb > 131072 triggers vector tiling
        np.random.seed(9000)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        # Allow small floating-point differences (GPU vs CPU precision)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        # Labels should match (same nearest neighbors)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_tiled_many_queries(self):
        """Force query tiling (nq > 512): compare Metal vs CPU."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 500, 600, 10  # nq > 512 triggers query tiling
        np.random.seed(9001)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        # Allow small floating-point differences (GPU vs CPU precision)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        # Labels should match (same nearest neighbors)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_tiled_single_column_tile(self):
        """Single column tile (nb < 131072) but multiple row tiles (nq=600)."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 50_000, 600, 10
        np.random.seed(9002)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        # Allow small floating-point differences (GPU vs CPU precision)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        # Labels should match (same nearest neighbors)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_bfknn_tiling_params_api(self):
        """Exercise params-based bfKnn_tiling API exposed through Metal bridge."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 64, 6000, 80, 10
        np.random.seed(9100)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        D_gpu = np.empty((nq, k), dtype=np.float32)
        I_gpu_i32 = np.empty((nq, k), dtype=np.int32)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_L2
        args.k = k
        args.dims = d
        args.vectors = faiss.swig_ptr(xb)
        args.vectorType = faiss.DistanceDataType_F32
        args.vectorsRowMajor = True
        args.numVectors = nb
        args.queries = faiss.swig_ptr(xq)
        args.queryType = faiss.DistanceDataType_F32
        args.queriesRowMajor = True
        args.numQueries = nq
        args.outDistances = faiss.swig_ptr(D_gpu)
        args.outIndicesType = faiss.IndicesDataType_I32
        args.outIndices = faiss.swig_ptr(I_gpu_i32)
        args.device = 0
        args.use_cuvs = False

        vectors_limit = d * 4 * 1200
        queries_limit = (d * 4 + k * (4 + 4)) * 24

        res = faiss.StandardGpuResources()
        faiss.bfKnn_tiling(res, args, vectors_limit, queries_limit)

        I_gpu = I_gpu_i32.astype(np.int64)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_bfknn_params_f16_supported(self):
        """Params API supports F16 vectors/queries (queries converted to F32 internally)."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 32, 3000, 40, 8
        np.random.seed(9200)
        xb = np.random.randn(nb, d).astype(np.float16)
        xq = np.random.randn(nq, d).astype(np.float16)
        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_L2
        args.k = k
        args.dims = d
        args.vectors = faiss.swig_ptr(xb)
        args.vectorType = faiss.DistanceDataType_F16
        args.vectorsRowMajor = True
        args.numVectors = nb
        args.queries = faiss.swig_ptr(xq)
        args.queryType = faiss.DistanceDataType_F16
        args.queriesRowMajor = True
        args.numQueries = nq
        args.outDistances = faiss.swig_ptr(D)
        args.outIndicesType = faiss.IndicesDataType_I64
        args.outIndices = faiss.swig_ptr(I)
        args.device = 0
        args.use_cuvs = False

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb.astype(np.float32))
        D_cpu, I_cpu = cpu_index.search(xq.astype(np.float32), k)

        res = faiss.StandardGpuResources()
        faiss.bfKnn(res, args)

        np.testing.assert_allclose(D, D_cpu, rtol=2e-3, atol=2e-3)
        np.testing.assert_array_equal(I, I_cpu)

    def test_bfknn_params_bf16_supported(self):
        """Params API accepts BF16 vectors/queries (converted to F32 internally)."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        if not hasattr(np, "bfloat16"):
            self.skipTest("numpy bfloat16 dtype not available")

        d, nb, nq, k = 16, 1200, 24, 6
        np.random.seed(9300)
        xb = np.random.randn(nb, d).astype(np.float32).astype(np.bfloat16)
        xq = np.random.randn(nq, d).astype(np.float32).astype(np.bfloat16)
        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_L2
        args.k = k
        args.dims = d
        args.vectors = faiss.swig_ptr(xb)
        args.vectorType = faiss.DistanceDataType_BF16
        args.vectorsRowMajor = True
        args.numVectors = nb
        args.queries = faiss.swig_ptr(xq)
        args.queryType = faiss.DistanceDataType_F32
        args.queriesRowMajor = True
        args.numQueries = nq
        args.outDistances = faiss.swig_ptr(D)
        args.outIndicesType = faiss.IndicesDataType_I64
        args.outIndices = faiss.swig_ptr(I)
        args.device = 0
        args.use_cuvs = False

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb.astype(np.float32))
        D_cpu, I_cpu = cpu_index.search(xq.astype(np.float32), k)

        res = faiss.StandardGpuResources()
        faiss.bfKnn(res, args)

        np.testing.assert_allclose(D, D_cpu, rtol=2e-3, atol=2e-3)
        np.testing.assert_array_equal(I, I_cpu)


if __name__ == "__main__":
    unittest.main()
