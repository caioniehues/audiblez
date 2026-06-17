"""Hermetic tests for the preflight checks (audiblez/doctor.py).

doctor.py imports only stdlib + audiblez.backends (no torch/kokoro/spacy at module
scope), so these run anywhere — the heavy stack is faked or its absence asserted.
"""
import os
import unittest
from unittest import mock
from tempfile import NamedTemporaryFile

from audiblez import doctor


class FindEspeakTest(unittest.TestCase):
    def test_env_override_existing_file_is_returned(self):
        with NamedTemporaryFile(suffix='.so') as f:
            with mock.patch.dict(os.environ, {'ESPEAK_LIBRARY': f.name}):
                self.assertEqual(doctor.find_espeak_library(), f.name)

    def test_env_override_missing_file_raises(self):
        with mock.patch.dict(os.environ, {'ESPEAK_LIBRARY': '/no/such/espeak.so'}):
            with self.assertRaises(RuntimeError):
                doctor.find_espeak_library()

    def test_linux_no_library_raises_with_apt_hint(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(doctor.platform, 'system', return_value='Linux'), \
             mock.patch.object(doctor, 'glob', return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                doctor.find_espeak_library()
            self.assertIn('apt install espeak-ng', str(ctx.exception))

    def test_linux_finds_first_match(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(doctor.platform, 'system', return_value='Linux'), \
             mock.patch.object(doctor, 'glob', side_effect=[['/usr/lib/x/libespeak-ng.so.1'], [], []]):
            self.assertEqual(doctor.find_espeak_library(), '/usr/lib/x/libespeak-ng.so.1')


class ChecksTest(unittest.TestCase):
    def test_ffmpeg_present_and_absent(self):
        with mock.patch.object(doctor.shutil, 'which', return_value='/usr/bin/ffmpeg'):
            self.assertEqual(doctor.check_ffmpeg().status, 'ok')
        with mock.patch.object(doctor.shutil, 'which', return_value=None):
            self.assertEqual(doctor.check_ffmpeg().status, 'fail')

    def test_ffprobe_absent_warns_not_fails(self):
        with mock.patch.object(doctor.shutil, 'which', return_value=None):
            self.assertEqual(doctor.check_ffprobe().status, 'warn')

    def test_spacy_missing_fails_model_missing_warns(self):
        with mock.patch.object(doctor.importlib.util, 'find_spec', return_value=None):
            self.assertEqual(doctor.check_spacy_model().status, 'fail')

        def only_spacy(name):
            return object() if name == 'spacy' else None
        with mock.patch.object(doctor.importlib.util, 'find_spec', side_effect=only_spacy):
            self.assertEqual(doctor.check_spacy_model().status, 'warn')

    def test_backend_unknown_fails(self):
        self.assertEqual(doctor.check_backend('bogus').status, 'fail')

    def test_torch_backend_without_torch_fails(self):
        with mock.patch.object(doctor.importlib.util, 'find_spec', return_value=None):
            self.assertEqual(doctor.check_backend('cpu').status, 'fail')

    def test_torch_backend_with_torch_available_ok(self):
        with mock.patch.object(doctor.importlib.util, 'find_spec', return_value=object()), \
             mock.patch.object(doctor.backends, 'available_backends', return_value=['cpu']):
            self.assertEqual(doctor.check_backend('cpu').status, 'ok')


class ReportTest(unittest.TestCase):
    def test_format_report_plain_has_no_ansi(self):
        results = [doctor.CheckResult('x', 'ok', 'fine'), doctor.CheckResult('y', 'fail', 'bad')]
        text = doctor.format_report(results, color=False)
        self.assertNotIn('\033', text)
        self.assertIn('x', text)
        self.assertIn('y', text)

    def test_run_doctor_returns_false_on_any_fail(self):
        lines = []
        with mock.patch.object(doctor, 'run_checks',
                               return_value=[doctor.CheckResult('x', 'fail', 'bad')]):
            ok = doctor.run_doctor(out=lines.append, color=False)
        self.assertFalse(ok)

    def test_run_doctor_returns_true_when_all_ok_or_warn(self):
        with mock.patch.object(doctor, 'run_checks',
                               return_value=[doctor.CheckResult('x', 'ok', 'fine'),
                                             doctor.CheckResult('y', 'warn', 'meh')]):
            self.assertTrue(doctor.run_doctor(out=lambda *_: None, color=False))


if __name__ == '__main__':
    unittest.main()
