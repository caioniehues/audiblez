"""Tests for sentence resilience + merge (core._retry / _synth_batch / find_chapter_wavs).

Guarded import so it skips without the heavy stack. synth is faked to raise on demand;
no model or audio is involved (a "segment" is the chunk string, except the silence
placeholder which is a real numpy array).
"""
import json
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
    import audiblez.core as core
    _ERR = None
except Exception as e:
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class RetryTest(unittest.TestCase):
    def test_returns_first_success_without_retry(self):
        fn = mock.MagicMock(return_value='ok')
        self.assertEqual(core._retry(fn, retries=2), 'ok')
        self.assertEqual(fn.call_count, 1)

    def test_retries_then_succeeds(self):
        fn = mock.MagicMock(side_effect=[RuntimeError('x'), 'ok'])
        self.assertEqual(core._retry(fn, retries=2), 'ok')
        self.assertEqual(fn.call_count, 2)

    def test_raises_after_exhausting_attempts(self):
        fn = mock.MagicMock(side_effect=RuntimeError('always'))
        with self.assertRaises(RuntimeError):
            core._retry(fn, retries=2)
        self.assertEqual(fn.call_count, 3)  # 1 + 2 retries


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class SynthBatchTest(unittest.TestCase):
    def test_happy_path_returns_batch_segments_no_dead_letter(self):
        synth = lambda t, sp: t.split('\n\n\n')
        with TemporaryDirectory() as tmp:
            dl = Path(tmp) / 'c.failed.jsonl'
            out = core._synth_batch(synth, ['a', 'b'], 1.0, retries=2, dead_letter_path=dl)
            self.assertEqual(out, ['a', 'b'])
            self.assertFalse(dl.exists())

    def test_batch_failure_falls_back_to_per_sentence(self):
        def synth(text, speed):
            if '\n\n\n' in text:
                raise RuntimeError('batch boom')
            return [text]
        out = core._synth_batch(synth, ['a', 'b'], 1.0, retries=1)
        self.assertEqual(out, ['a', 'b'])

    def test_persistent_sentence_failure_splices_silence_and_dead_letters(self):
        def synth(text, speed):
            if '\n\n\n' in text or text == 'bad':
                raise RuntimeError('boom')
            return [text]
        with TemporaryDirectory() as tmp:
            dl = Path(tmp) / 'c.failed.jsonl'
            out = core._synth_batch(synth, ['good', 'bad'], 1.0, retries=1,
                                    chapter_label='chapter 3', dead_letter_path=dl)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0], 'good')
            self.assertIsInstance(out[1], np.ndarray)   # silence placeholder
            self.assertTrue((out[1] == 0).all())
            self.assertGreater(len(out[1]), 0)
            lines = dl.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec['text'], 'bad')
            self.assertEqual(rec['chapter'], 'chapter 3')


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class FindChapterWavsTest(unittest.TestCase):
    def test_orders_numerically_not_lexicographically(self):
        with TemporaryDirectory() as tmp:
            for i, x in [(1, 'intro'), (2, 'mid'), (10, 'end')]:
                (Path(tmp) / f'book_chapter_{i}_af_sky_{x}.wav').write_bytes(b'x')
            # a different voice / book must not match
            (Path(tmp) / 'book_chapter_3_bf_emma_z.wav').write_bytes(b'x')
            found = core.find_chapter_wavs('book.epub', 'af_sky', tmp)
            names = [p.name for p in found]
            self.assertEqual(names, ['book_chapter_1_af_sky_intro.wav',
                                     'book_chapter_2_af_sky_mid.wav',
                                     'book_chapter_10_af_sky_end.wav'])


if __name__ == '__main__':
    unittest.main()
