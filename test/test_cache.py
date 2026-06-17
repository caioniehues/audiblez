"""Tests for the opt-in synth cache (audiblez/cache.py).

Pure numpy/hashlib/json module, so these run anywhere (no skip guard).
"""
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
from tempfile import TemporaryDirectory

from audiblez import cache

try:
    import audiblez.core as core
    _CORE_ERR = None
except Exception as e:
    _CORE_ERR = e


_FIELDS = dict(engine='torch', repo_id='hexgrad/Kokoro-82M', voice='af_sky',
               speed=1.0, max_sentence_length=400, spacy_version='3.8.3')


class MakeKeyTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(cache.make_key(text='hello', **_FIELDS),
                         cache.make_key(text='hello', **_FIELDS))

    def test_sensitive_to_every_field(self):
        base = cache.make_key(text='hello', **_FIELDS)
        self.assertNotEqual(base, cache.make_key(text='world', **_FIELDS))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'voice': 'am_adam'}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'speed': 1.5}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'engine': 'mlx'}))
        # repo_id differs by quantization between backends -> must change the key
        self.assertNotEqual(base, cache.make_key(text='hello',
                                                 **{**_FIELDS, 'repo_id': 'mlx-community/Kokoro-82M-bf16'}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'max_sentence_length': 200}))

    def test_version_changes_key(self):
        k1 = cache.make_key(text='hello', **_FIELDS)
        old = cache.CACHE_VERSION
        try:
            cache.CACHE_VERSION = old + 1
            self.assertNotEqual(k1, cache.make_key(text='hello', **_FIELDS))
        finally:
            cache.CACHE_VERSION = old


class SynthCacheTest(unittest.TestCase):
    def test_miss_then_put_then_hit(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            key = cache.make_key(text='hello', **_FIELDS)
            self.assertIsNone(c.get(key))      # miss
            self.assertEqual(c.misses, 1)
            audio = np.arange(10, dtype=np.float32)
            c.put(key, audio)
            got = c.get(key)                   # hit
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got, audio)
            self.assertEqual(c.hits, 1)

    def test_corrupt_entry_is_a_miss(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            key = cache.make_key(text='x', **_FIELDS)
            (c._path(key)).write_bytes(b'not a real npy')
            self.assertIsNone(c.get(key))
            self.assertEqual(c.misses, 1)

    def test_stats_string(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            c.hits, c.misses = 3, 1
            self.assertIn('75% hit rate', c.stats())


@unittest.skipIf(_CORE_ERR is not None, f"audiblez.core unavailable: {_CORE_ERR}")
class GenAudioSegmentsCacheTest(unittest.TestCase):
    def _fake_nlp(self):
        return lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('|')])

    def test_second_run_hits_cache_and_skips_synth(self):
        fields = dict(_FIELDS)
        synth = mock.MagicMock(side_effect=lambda t, sp: [np.ones(3, dtype=np.float32)])
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                out1 = core.gen_audio_segments(synth, 'one|two', voice='af_sky', speed=1.0,
                                               cache=c, cache_key_fields=fields)
            self.assertEqual(synth.call_count, 2)   # two sentences synthesized (misses)
            self.assertEqual((c.hits, c.misses), (0, 2))

            synth.reset_mock()
            c2 = cache.SynthCache(tmp)              # fresh cache obj, same dir
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                out2 = core.gen_audio_segments(synth, 'one|two', voice='af_sky', speed=1.0,
                                               cache=c2, cache_key_fields=fields)
            self.assertEqual(synth.call_count, 0)   # all served from cache
            self.assertEqual((c2.hits, c2.misses), (2, 0))
            for a, b in zip(out1, out2):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
