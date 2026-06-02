"""Hermetic tests for the pure logic in build_dataset.py (segmentation, manifests, text).

No ML deps needed — build_dataset only imports numpy + soundfile at module scope (the
heavy whisperx/demucs/speechbrain imports are lazy). Run from the repo root:

    python -m unittest discover -s tools/voice_dataset -p 'test_*.py' -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as B  # noqa: E402


def W(word, start, end, speaker='S0'):
    return {'word': word, 'start': start, 'end': end, 'speaker': speaker}


class EndsSentenceTest(unittest.TestCase):
    def test_plain_terminals(self):
        self.assertTrue(B.ends_sentence('done.'))
        self.assertTrue(B.ends_sentence('really?'))
        self.assertTrue(B.ends_sentence('wow!'))
        self.assertFalse(B.ends_sentence('word'))

    def test_trailing_quotes_and_brackets(self):
        self.assertTrue(B.ends_sentence('go!"'))
        self.assertTrue(B.ends_sentence('really?”'))
        self.assertTrue(B.ends_sentence("(yes.)"))
        self.assertFalse(B.ends_sentence('mid"'))


class NormalizeTextTest(unittest.TestCase):
    def test_adds_terminal(self):
        self.assertEqual(B.normalize_text('hello world'), 'hello world.')

    def test_collapses_whitespace(self):
        self.assertEqual(B.normalize_text('a   b\n c'), 'a b c.')

    def test_keeps_existing_terminal_with_quote(self):
        self.assertEqual(B.normalize_text('go!"'), 'go!"')


class SegmentWordsTest(unittest.TestCase):
    def seg(self, words, min_dur=1.0, max_dur=15.0, min_chars=6, max_gap=1.0):
        return B.segment_words(words, min_dur, max_dur, min_chars, max_gap)

    def test_every_clip_ends_a_sentence(self):
        words = [W('Hi.', 0, 0.6), W('A', 0.6, 0.9), W('B', 0.9, 1.2),
                 W('ok?', 1.2, 1.8), W('frag', 1.8, 2.2)]
        segs = self.seg(words, min_dur=0.5, min_chars=3)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_punctuationless_runon_dropped(self):
        runon = [W('word', i * 0.5, i * 0.5 + 0.5) for i in range(40)]
        self.assertEqual(self.seg(runon, max_dur=6.0), [])

    def test_gap_breaks_into_separate_clips(self):
        words = [W('First', 0, 0.5), W('part.', 0.5, 1.2),
                 W('Second', 6.5, 7.0), W('part.', 7.0, 7.8)]
        segs = self.seg(words)
        self.assertEqual(len(segs), 2)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_mid_sentence_fragment_before_gap_dropped(self):
        words = [W('A', 0, 0.5), W('mid', 0.5, 1.0), W('fragment', 1.0, 1.6), W('Done.', 7.0, 7.6)]
        segs = self.seg(words)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_overflow_cut_respects_max_dur_and_sentence(self):
        long = [W('word.' if i % 5 == 4 else 'word', i * 0.5, i * 0.5 + 0.5) for i in range(40)]
        segs = self.seg(long, max_dur=6.0)
        self.assertTrue(all((e - s) <= 6.01 for s, e, _ in segs))
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_closing_quote_clip_kept(self):
        words = [W('He', 0, 0.4), W('said', 0.4, 0.8), W('go!"', 0.8, 1.4)]
        segs = self.seg(words)
        self.assertEqual(len(segs), 1)
        self.assertTrue(segs[0][2].endswith('go!"'))


class SelectNarratorTest(unittest.TestCase):
    def test_dominant_speaker_fallback(self):
        words = [W('a', 0, 3, 'S0'), W('b', 3, 3.5, 'S1'), W('c', 3.5, 4, 'S1')]
        spk, _info = B.select_narrator(words, None, 24000, None, None, 0.5)
        self.assertEqual(spk, 'S0')

    def test_no_words_returns_none(self):
        spk, _ = B.select_narrator([], None, 24000, None, None, 0.5)
        self.assertIsNone(spk)


class ManifestsTest(unittest.TestCase):
    def test_three_formats(self):
        recs = [{'id': f'host-{i:06d}', 'raw': f'Line {i}', 'norm': f'Line {i}.'} for i in range(20)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            B.write_manifests(out, recs, 'host', val_fraction=0.1)
            lj = (out / 'metadata.csv').read_text().splitlines()
            self.assertEqual(lj[0], 'host-000000|Line 0|Line 0.')      # no header, no .wav
            sty = (out / 'train_list.txt').read_text().splitlines()
            val = (out / 'val_list.txt').read_text().splitlines()
            self.assertTrue(sty[0].endswith('|0') and '.wav|' in sty[0])  # .wav + int speaker
            self.assertEqual((len(val), len(sty)), (2, 18))               # ~10% val split
            xt = (out / 'metadata_xtts.csv').read_text().splitlines()
            self.assertEqual(xt[0], 'audio_file|text|speaker_name')       # header
            self.assertTrue(xt[1].startswith('wavs/host-'))


if __name__ == '__main__':
    unittest.main()
