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


if __name__ == "__main__":
    unittest.main()
