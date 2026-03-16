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
        """Same as above but pass GpuClonerOptions for Metal cloner parity."""
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

    def test_bfknn_params_column_major_f32(self):
        """Params API accepts column-major vectors/queries via row-major flags."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 24, 2500, 32, 6
        np.random.seed(9400)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        xb_col = np.asfortranarray(xb)
        xq_col = np.asfortranarray(xq)

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_L2
        args.k = k
        args.dims = d
        args.vectors = faiss.swig_ptr(xb_col)
        args.vectorType = faiss.DistanceDataType_F32
        args.vectorsRowMajor = False
        args.numVectors = nb
        args.queries = faiss.swig_ptr(xq_col)
        args.queryType = faiss.DistanceDataType_F32
        args.queriesRowMajor = False
        args.numQueries = nq
        args.outDistances = faiss.swig_ptr(D)
        args.outIndicesType = faiss.IndicesDataType_I64
        args.outIndices = faiss.swig_ptr(I)
        args.device = 0
        args.use_cuvs = False

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        res = faiss.StandardGpuResources()
        faiss.bfKnn_tiling(
            res,
            args,
            d * 4 * 800,
            (d * 4 + k * (4 + 8)) * 16,
        )

        np.testing.assert_allclose(D, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I, I_cpu)

    def test_bfknn_params_vector_norms_l2(self):
        """L2 params path accepts precomputed vector norms."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 24, 2200, 28, 6
        np.random.seed(9500)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)
        xb_norms = np.sum(xb * xb, axis=1).astype(np.float32)

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_L2
        args.k = k
        args.dims = d
        args.vectors = faiss.swig_ptr(xb)
        args.vectorType = faiss.DistanceDataType_F32
        args.vectorsRowMajor = True
        args.numVectors = nb
        args.vectorNorms = faiss.swig_ptr(xb_norms)
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
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        res = faiss.StandardGpuResources()
        faiss.bfKnn_tiling(
            res,
            args,
            d * 4 * 900,
            (d * 4 + k * (4 + 8)) * 16,
        )

        np.testing.assert_allclose(D, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I, I_cpu)

    def test_pairwise_distance_gpu_all_pairs(self):
        """k == -1 path returns full pairwise matrix."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq = 20, 900, 22
        np.random.seed(9600)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu = np.sum((xq[:, None, :] - xb[None, :, :]) ** 2, axis=2, dtype=np.float32)

        res = faiss.StandardGpuResources()
        D = faiss.pairwise_distance_gpu(res, xq, xb, metric=faiss.METRIC_L2)
        np.testing.assert_allclose(D, cpu, rtol=1e-5, atol=1e-5)

    def test_knn_gpu_l1_metric(self):
        """knn_gpu supports non-L2/IP metrics via params fallback path."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 16, 1200, 24, 5
        np.random.seed(9700)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        res = faiss.StandardGpuResources()
        D, I = faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_L1)
        np.testing.assert_allclose(D, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I, I_cpu)

    def test_pairwise_distance_gpu_l1_metric(self):
        """pairwise_distance_gpu supports non-L2/IP metrics via params fallback path."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq = 14, 700, 18
        np.random.seed(9800)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu = np.sum(np.abs(xq[:, None, :] - xb[None, :, :]), axis=2, dtype=np.float32)

        res = faiss.StandardGpuResources()
        D = faiss.pairwise_distance_gpu(res, xq, xb, metric=faiss.METRIC_L1)
        np.testing.assert_allclose(D, cpu, rtol=1e-5, atol=1e-5)

    def test_bfknn_params_lp_metric(self):
        """Params API supports METRIC_Lp with metricArg via fallback path."""
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

        d, nb, nq, k = 12, 900, 20, 4
        p = 3.0
        np.random.seed(9900)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlat(d, faiss.METRIC_Lp)
        cpu_index.metric_arg = p
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        args = faiss.GpuDistanceParams()
        args.metric = faiss.METRIC_Lp
        args.metricArg = p
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
        args.outDistances = faiss.swig_ptr(D)
        args.outIndicesType = faiss.IndicesDataType_I64
        args.outIndices = faiss.swig_ptr(I)
        args.device = 0
        args.use_cuvs = False

        res = faiss.StandardGpuResources()
        faiss.bfKnn(res, args)
        np.testing.assert_allclose(D, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I, I_cpu)


if __name__ == "__main__":
    unittest.main()
