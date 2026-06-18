"""Guarded tests for core.build_synthesizer.

Importing audiblez.core needs the TTS stack, so these skip cleanly when it is
absent (and run in CI where deps are installed). No model/network is used — the
engines are faked.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_preset_blend_resolves_lang_and_blend_string(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            synth = core.build_synthesizer('af_warm', 'cpu')  # Heart + Bella
            self.assertEqual(kp.call_args.kwargs.get('lang_code'), 'a')
            synth('hello world', 1.0)
            # The comma blend string is what both Kokoro engines average.
            self.assertEqual(fake_pipeline.call_args.kwargs.get('voice'), 'af_heart,af_bella')

    def test_weighted_blend_passes_repetition_string(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline):
            synth = core.build_synthesizer('af_bella:60,af_heart:40', 'cpu')
            synth('hello world', 1.0)
            self.assertEqual(fake_pipeline.call_args.kwargs.get('voice'),
                             'af_bella,af_bella,af_bella,af_heart,af_heart')

    def test_unknown_voice_raises_valueerror(self):
        with self.assertRaises(ValueError):
            core.build_synthesizer('af_nonexistent', 'cpu')

    def test_unknown_backend_raises_valueerror(self):
        with self.assertRaises(ValueError):
            core.build_synthesizer('af_sky', 'bogus')

    def test_mlx_unavailable_raises_clear_runtimeerror(self):
        with mock.patch.object(backends, '_mlx_importable', return_value=False):
            with self.assertRaises(RuntimeError):
                core.build_synthesizer('af_sky', 'mlx')

    def test_torch_synth_coerces_tensor_segment_to_numpy(self):
        # Regression: kokoro may yield a torch.Tensor (possibly on GPU). build_synthesizer's closure must
        # to_numpy() each segment so np.concatenate/soundfile.write downstream work. Fake the pipeline to
        # emit a torch tensor and assert the closure returns a concatenatable numpy array.
        import torch
        fake_audio = torch.zeros(4, dtype=torch.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline):
            synth = core.build_synthesizer('af_sky', 'cpu')
            out = synth('hello world', 1.0)
        self.assertTrue(out and all(isinstance(x, np.ndarray) for x in out))
        # np.concatenate is what core actually does with the segments; it must not raise on the coerced output.
        self.assertEqual(np.concatenate(out).shape, (4,))


def _drive_main(d, chapters, voice='af_heart', shutil_which=None, post_event=None,
                gen_segments=None):
    """Run core.main() over `chapters` with all heavy I/O mocked (no epub/spaCy/ffmpeg/disk).

    `shutil_which` is the return of shutil.which (None => ffmpeg absent). `gen_segments`, if given,
    is used as gen_audio_segments' side_effect/return so callers can inspect the text it receives.
    """
    if gen_segments is None:
        gen_segments = mock.Mock(return_value=[np.zeros(4, dtype=np.float32)])
    with mock.patch.object(core, 'load_spacy'), \
         mock.patch.object(core, 'epub') as ep, \
         mock.patch.object(core, 'extract_book_metadata', return_value=('Title', 'Author')), \
         mock.patch.object(core, 'find_cover', return_value=None), \
         mock.patch.object(core, 'find_document_chapters_and_extract_texts', return_value=chapters), \
         mock.patch.object(core, 'find_good_chapters', return_value=chapters), \
         mock.patch.object(core, 'set_espeak_library'), \
         mock.patch.object(core, 'build_synthesizer', return_value=lambda *a, **k: []), \
         mock.patch.object(core, 'gen_audio_segments', gen_segments), \
         mock.patch.object(core, 'shutil') as sh, \
         mock.patch.object(core, 'soundfile') as sf:
        ep.read_epub.return_value = object()
        sh.which.return_value = shutil_which
        sf.write.return_value = None
        core.main('book.epub', voice, False, 1.0, output_folder=d, post_event=post_event)
    return gen_segments


def _chapter(text, index=0, name='Text/Chap01.xhtml'):
    return SimpleNamespace(extracted_text=text, chapter_index=index, get_name=lambda: name)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MainTerminalEventTest(unittest.TestCase):
    def test_core_finished_emitted_when_ffmpeg_absent(self):
        # GUI-hang regression: with ffmpeg missing (shutil.which -> None) main() still posts CORE_FINISHED,
        # even for a short/empty chapter set that produces no wavs — otherwise the GUI waits on it forever.
        events = []
        with tempfile.TemporaryDirectory() as d:
            _drive_main(d, [_chapter('short')],  # < 10 chars -> skipped, no wavs synthesized
                        shutil_which=None, post_event=lambda name, **kw: events.append(name))
        self.assertIn('CORE_FINISHED', events)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class IntroOnResumeTest(unittest.TestCase):
    def test_intro_not_reprepended_when_chapter1_already_exists(self):
        # Resume regression: an already-written chapter-1 wav (text len >= 10) already contains the prepended
        # 'Title – Author' intro, so the resumed synth of chapter 2 must NOT prepend it again.
        ch1 = _chapter('Chapter one body, definitely longer than ten characters.', index=0,
                       name='Text/Chap01.xhtml')
        ch2 = _chapter('Chapter two body, also clearly longer than ten characters.', index=1,
                       name='Text/Chap02.xhtml')
        seen = []
        gen = mock.Mock(side_effect=lambda synth, text, *a, **k: (seen.append(text),
                                                                  [np.zeros(4, dtype=np.float32)])[1])
        with tempfile.TemporaryDirectory() as d:
            voice_tag = 'af_heart'
            # Pre-create chapter 1's wav at the exact path main() computes, so it is skipped (resume).
            (Path(d) / f'book_chapter_1_{voice_tag}_Text_Chap01.xhtml.wav').write_bytes(b'RIFF')
            # The stub wav can't pass the real validity/render-signature gate (no audio, no .sig),
            # so force it 'valid & complete' to isolate the intro-on-resume behavior under test.
            with mock.patch.object(core, 'is_valid_chapter_wav', return_value=True), \
                 mock.patch.object(core, '_chapter_is_complete', return_value=True):
                _drive_main(d, [ch1, ch2], gen_segments=gen)

        # Only chapter 2 was synthesized (chapter 1 skipped as already-existing).
        self.assertEqual(len(seen), 1)
        synthesized = seen[0]
        self.assertNotIn('Title – Author', synthesized)  # intro was consumed by the skipped chapter 1
        self.assertTrue(synthesized.startswith('Chapter two body'))


if __name__ == '__main__':
    unittest.main()
