"""Hermetic tests for the backend registry + detection (audiblez/backends.py).

backends.py imports only stdlib at module scope (torch is imported lazily inside
available_backends), so these run anywhere — torch is faked via sys.modules.
"""
import unittest
from unittest import mock

from audiblez import backends


def _fake_torch(cuda=False, hip=None, cuda_ver=None, mps=False):
    t = mock.MagicMock()
    t.cuda.is_available.return_value = cuda
    t.version.hip = hip
    t.version.cuda = cuda_ver
    t.backends.mps.is_available.return_value = mps
    return t


class BackendTableTest(unittest.TestCase):
    def test_ids_and_order(self):
        self.assertEqual(backends.BACKEND_IDS, ('cpu', 'cuda', 'rocm', 'mps', 'mlx'))

    def test_device_mapping(self):
        # cuda and rocm both ride the torch.cuda API -> same torch device
        self.assertEqual(backends.BACKENDS['cuda'].torch_device, 'cuda')
        self.assertEqual(backends.BACKENDS['rocm'].torch_device, 'cuda')
        self.assertEqual(backends.BACKENDS['mps'].torch_device, 'mps')
        self.assertEqual(backends.BACKENDS['cpu'].torch_device, 'cpu')
        # mlx is a separate engine, no torch device
        self.assertEqual(backends.BACKENDS['mlx'].engine, 'mlx')
        self.assertIsNone(backends.BACKENDS['mlx'].torch_device)

    def test_is_gpu(self):
        for gpu in ('cuda', 'rocm', 'mps', 'mlx'):
            self.assertTrue(backends.is_gpu(gpu))
        self.assertFalse(backends.is_gpu('cpu'))


class DetectionTest(unittest.TestCase):
    def _avail(self, torch_obj, mlx=False):
        with mock.patch.dict('sys.modules', {'torch': torch_obj}), \
             mock.patch.object(backends, '_mlx_importable', return_value=mlx):
            return backends.available_backends()

    def test_cpu_only(self):
        self.assertEqual(self._avail(_fake_torch()), ['cpu'])

    def test_nvidia(self):
        self.assertEqual(self._avail(_fake_torch(cuda=True, cuda_ver='12.1')), ['cpu', 'cuda'])

    def test_amd_rocm_never_cuda(self):
        # A HIP build reports rocm, never cuda.
        self.assertEqual(self._avail(_fake_torch(cuda=True, hip='6.2')), ['cpu', 'rocm'])

    def test_apple_silicon(self):
        self.assertEqual(self._avail(_fake_torch(mps=True), mlx=True), ['cpu', 'mps', 'mlx'])

    def test_torch_import_failure_degrades_to_cpu(self):
        boom = mock.MagicMock()
        boom.cuda.is_available.side_effect = RuntimeError('no torch')
        self.assertEqual(self._avail(boom), ['cpu'])

    def test_default_prefers_accelerator(self):
        cases = {
            ('cpu',): 'cpu',
            ('cpu', 'cuda'): 'cuda',
            ('cpu', 'rocm'): 'rocm',
            ('cpu', 'mps', 'mlx'): 'mps',   # torch MPS preferred over mlx as the safe default
        }
        for avail, expected in cases.items():
            with mock.patch.object(backends, 'available_backends', return_value=list(avail)):
                self.assertEqual(backends.default_backend(), expected)


if __name__ == '__main__':
    unittest.main()
