"""Guarded tests for core.build_synthesizer.

Importing audiblez.core needs the TTS stack, so these skip cleanly when it is
absent (and run in CI where deps are installed). No model/network is used — the
engines are faked.
"""
import unittest
from unittest import mock

try:
    import numpy as np
    import audiblez.core as core
    from audiblez import backends
    _ERR = None
except Exception as e:  # heavy deps (torch/kokoro/...) may be absent
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class BuildSynthesizerTest(unittest.TestCase):
    def test_torch_path_passes_device_and_returns_numpy(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            synth = core.build_synthesizer('af_sky', 'mps')
            self.assertEqual(kp.call_args.kwargs.get('device'), 'mps')  # device threaded through
            out = synth('hello world', 1.0)
            self.assertIsInstance(out, list)
            self.assertTrue(out and all(isinstance(x, np.ndarray) for x in out))

    def test_rocm_maps_to_cuda_device(self):
        fake_pipeline = mock.MagicMock(return_value=[])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            core.build_synthesizer('af_sky', 'rocm')
            self.assertEqual(kp.call_args.kwargs.get('device'), 'cuda')

    def test_unknown_backend_raises_valueerror(self):
        with self.assertRaises(ValueError):
            core.build_synthesizer('af_sky', 'bogus')

    def test_mlx_unavailable_raises_clear_runtimeerror(self):
        with mock.patch.object(backends, '_mlx_importable', return_value=False):
            with self.assertRaises(RuntimeError):
                core.build_synthesizer('af_sky', 'mlx')


if __name__ == '__main__':
    unittest.main()
