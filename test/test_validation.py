"""Tests for validate-before-skip (core.is_valid_chapter_wav).

Guarded import so it skips when the heavy stack (torch/kokoro/spacy) is absent.
The decision logic is exercised by faking probe_duration/soundfile, so no real
audio model, ffprobe, or wav encoder is needed.
"""
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import audiblez.core as core
    _ERR = None
except Exception as e:  # heavy deps may be absent
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class ValidWavTest(unittest.TestCase):
    def _file(self, tmp, size):
        p = Path(tmp) / 'c.wav'
        p.write_bytes(b'\0' * size)
        return p

    def test_missing_file_is_invalid(self):
        self.assertFalse(core.is_valid_chapter_wav('/no/such/file.wav', 100))

    def test_zero_byte_file_is_invalid(self):
        with TemporaryDirectory() as tmp:
            p = self._file(tmp, 0)
            self.assertFalse(core.is_valid_chapter_wav(p, 100))

    def test_plausible_duration_is_valid(self):
        with TemporaryDirectory() as tmp:
            p = self._file(tmp, 2048)
            # 1000 chars / 30 cps = 33s minimum *0.5 = ~16.6s floor; 60s clears it.
            with mock.patch.object(core, 'probe_duration', return_value=60.0):
                self.assertTrue(core.is_valid_chapter_wav(p, 1000))

    def test_truncated_duration_is_invalid(self):
        with TemporaryDirectory() as tmp:
            p = self._file(tmp, 2048)
            # 3000 chars needs >=50s; a 2s file is a crash truncation.
            with mock.patch.object(core, 'probe_duration', return_value=2.0):
                self.assertFalse(core.is_valid_chapter_wav(p, 3000))

    def test_max_sentences_skips_length_check(self):
        with TemporaryDirectory() as tmp:
            p = self._file(tmp, 2048)
            # expected_text_len None -> any readable, positive-duration file is fine.
            with mock.patch.object(core, 'probe_duration', return_value=1.0):
                self.assertTrue(core.is_valid_chapter_wav(p, None))

    def test_falls_back_to_soundfile_when_ffprobe_absent(self):
        with TemporaryDirectory() as tmp:
            p = self._file(tmp, 2048)
            info = mock.MagicMock(duration=60.0)
            with mock.patch.object(core, 'probe_duration', return_value=None), \
                 mock.patch.object(core.soundfile, 'info', return_value=info):
                self.assertTrue(core.is_valid_chapter_wav(p, 1000))

    def test_size_floor_when_no_duration_available(self):
        with TemporaryDirectory() as tmp:
            big = self._file(tmp, 5000)
            small = Path(tmp) / 'small.wav'
            small.write_bytes(b'\0' * 100)
            with mock.patch.object(core, 'probe_duration', return_value=None), \
                 mock.patch.object(core.soundfile, 'info', side_effect=RuntimeError('no sf')):
                self.assertTrue(core.is_valid_chapter_wav(big, 1000))
                self.assertFalse(core.is_valid_chapter_wav(small, 1000))


if __name__ == '__main__':
    unittest.main()
