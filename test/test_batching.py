"""Tests for sentence batching (core.pack_sentences + gen_audio_segments).

Guarded import so it skips without the heavy stack. The synth and spaCy pipeline
are faked, so no model/audio is needed: a "segment" is just the chunk string, which
lets us assert ordering and call count.
"""
import unittest
from unittest import mock
from types import SimpleNamespace

try:
    import audiblez.core as core
    _ERR = None
except Exception as e:
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class PackSentencesTest(unittest.TestCase):
    def test_packs_under_cap_into_fewer_batches(self):
        sents = ['a' * 100] * 10  # 1000 chars total
        batches = core.pack_sentences(sents, batch_max_chars=300)
        self.assertTrue(all(sum(len(s) for s in b) <= 300 for b in batches))
        self.assertLess(len(batches), len(sents))  # actually batched
        self.assertEqual([s for b in batches for s in b], sents)  # order preserved

    def test_oversized_sentence_is_its_own_batch(self):
        sents = ['short', 'x' * 5000, 'also short']
        batches = core.pack_sentences(sents, batch_max_chars=1000)
        self.assertIn(['x' * 5000], batches)
        self.assertEqual([s for b in batches for s in b], sents)

    def test_tiny_cap_is_one_sentence_per_batch(self):
        sents = ['one', 'two', 'three']
        batches = core.pack_sentences(sents, batch_max_chars=1)
        self.assertEqual(batches, [['one'], ['two'], ['three']])

    def test_empty(self):
        self.assertEqual(core.pack_sentences([], 1000), [])


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class GenAudioSegmentsBatchingTest(unittest.TestCase):
    def _fake_nlp(self):
        # Sentences are pipe-separated in the input text.
        return lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('|')])

    def test_batched_call_preserves_order_and_count(self):
        # fake synth: Kokoro re-splits on \n\n\n -> one "segment" (the chunk str) each.
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three', voice='af_sky', speed=1.0,
                                          batch_max_chars=1000)
        self.assertEqual(out, ['one', 'two', 'three'])   # order preserved
        self.assertEqual(synth.call_count, 1)            # one batched call

    def test_tiny_cap_falls_back_to_per_sentence_calls(self):
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three', voice='af_sky', speed=1.0,
                                          batch_max_chars=1)
        self.assertEqual(out, ['one', 'two', 'three'])
        self.assertEqual(synth.call_count, 3)            # per-sentence

    def test_max_sentences_caps_output(self):
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three|four', voice='af_sky', speed=1.0,
                                          max_sentences=2, batch_max_chars=1000)
        self.assertEqual(out, ['one', 'two'])


if __name__ == '__main__':
    unittest.main()
