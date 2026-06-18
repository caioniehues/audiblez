# -*- coding: utf-8 -*-
import argparse
import os
import sys

from audiblez.voices import voices, available_voices_str
from audiblez import backends


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -v af_sky -b mlx\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              available_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', nargs='?', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help='Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help='Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-b', '--backend', choices=backends.BACKEND_IDS, default=None,
                        help='Narration backend: cpu, cuda (NVIDIA), rocm (AMD), mps (Apple Silicon), '
                             'mlx (Apple Silicon native). Default: cpu.')
    parser.add_argument('-c', '--cuda', default=False, action='store_true',
                        help=argparse.SUPPRESS)  # deprecated alias for --backend cuda
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--doctor', default=False, action='store_true',
                        help='Run preflight checks (ffmpeg, espeak-ng, spaCy, selected backend) and exit')
    parser.add_argument('--deep', default=False, action='store_true',
                        help='With --doctor: also load the model and synthesize one word (slow)')
    parser.add_argument('--merge', default=False, action='store_true',
                        help='Assemble already-synthesized chapter wavs into an m4b and exit '
                             '(recover a playable audiobook from an interrupted run)')
    parser.add_argument('--trailer', default=False, action='store_true',
                        help='Render a short trailer.wav sampling the opening of each chapter '
                             'and exit (audition voice + chapter detection before a full run)')
    parser.add_argument('--seed-lexicon', dest='seed_lexicon', default=False, action='store_true',
                        help='Write a <book>.lexicon.json of candidate names/acronyms to edit, '
                             'then exit; applied as pronunciation overrides on the next run')
    parser.add_argument('--cache', default=False, action='store_true',
                        help='Cache synthesized sentences under <output>/.audiblez_cache and '
                             'reuse them on re-runs (opt-in; cache only, no resume)')
    parser.add_argument('--cache-clear', dest='cache_clear', default=False, action='store_true',
                        help='Delete the sentence cache under <output>/.audiblez_cache and exit')
    parser.add_argument('--tune', default=False, action='store_true',
                        help='Enable GPU GEMM autotuning (PyTorch TunableOp) for cuda/rocm. '
                             'First run is slower while it tunes; results are cached under '
                             '<output> and reused. No-op on cpu/mlx.')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32',
                        help='GPU compute precision via autocast (cuda/rocm only). fp16/bf16 are '
                             'faster but change the waveform — audition before a full run. '
                             'Default: fp32. No-op on cpu/mlx.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # --doctor checks the backend the real run will actually use — for a bare invocation
    # that is now the auto-selected backend, not a silent CPU default — and runs before
    # importing the heavy TTS stack so it works on a broken install.
    if args.doctor:
        from audiblez import doctor
        requested = (args.backend if args.backend is not None
                     else ('cuda' if args.cuda else backends.default_backend()))
        sys.exit(0 if doctor.run_doctor(backend=requested, voice=args.voice, deep=args.deep,
                                        tune=args.tune, output_folder=args.output,
                                        precision=args.precision) else 1)

    # --cache-clear wipes the sentence cache; needs no epub/backend/model.
    if args.cache_clear:
        from audiblez.cache import SynthCache
        cache_dir = os.path.join(args.output, '.audiblez_cache')
        print(f'Cleared {SynthCache(cache_dir).clear()} cached sentence(s) from {cache_dir}.')
        sys.exit(0)

    if not args.epub_file_path:
        parser.error('the following arguments are required: epub_file_path (or use --doctor)')

    # Ensure the output folder exists before any subcommand writes into it (--merge,
    # --seed-lexicon, --trailer all run before main(), which is what otherwise creates it).
    if args.output != '.':
        os.makedirs(args.output, exist_ok=True)

    # --merge needs no backend/model: just stitch existing chapter wavs into an m4b.
    if args.merge:
        from audiblez.core import merge_chapters
        result = merge_chapters(args.epub_file_path, args.voice, args.output)
        sys.exit(0 if result else 1)

    # --seed-lexicon writes editable pronunciation candidates; no backend/model needed.
    if args.seed_lexicon:
        from ebooklib import epub
        from audiblez import lexicon
        from audiblez.core import find_document_chapters_and_extract_texts
        chapters = find_document_chapters_and_extract_texts(epub.read_epub(args.epub_file_path))
        text = '\n'.join(c.extracted_text for c in chapters)
        path = lexicon.lexicon_path(args.epub_file_path, args.output)
        merged = lexicon.build_seed_lexicon(text)
        merged.update(lexicon.load_lexicon(path))  # never clobber existing user edits
        lexicon.save_lexicon(path, merged)
        print(f'Wrote {len(merged)} candidate term(s) to {path}.')
        print('Edit the JSON values to set pronunciations (respellings); they apply on the next run.')
        sys.exit(0)

    avail = backends.available_backends()
    if args.backend is not None:
        backend = args.backend
        if args.cuda and backend not in ('cuda', 'rocm'):
            parser.error(f'--cuda conflicts with --backend {backend}')
    elif args.cuda:
        print('Note: --cuda is deprecated; use --backend cuda')
        # Legacy intent = "use the torch GPU". On a ROCm build that GPU is exposed as 'rocm'.
        backend = 'cuda' if 'cuda' in avail else ('rocm' if 'rocm' in avail else 'cpu')
        if backend == 'cpu':
            print('CUDA GPU not available. Defaulting to CPU')
    else:
        # Auto-select the best available backend (GPU when present), falling back to CPU.
        # Probe the GPU with a real kernel: torch.cuda.is_available() can be True while the
        # device faults on first use (e.g. RDNA3 without HSA_OVERRIDE_GFX_VERSION), which
        # used to abort the run or emit silent audio. We only auto-pick a GPU that works.
        backend = backends.default_backend()
        if backend != 'cpu':
            if backends.gpu_works(backend):
                print(f'Auto-selected {backend} backend (pass -b cpu to force CPU)')
            else:
                print(f'\033[93mAuto-selected {backend} GPU failed a test kernel; falling back '
                      f'to CPU. Pass -b {backend} to force it (e.g. after setting '
                      'HSA_OVERRIDE_GFX_VERSION).\033[0m')
                backend = 'cpu'

    if backend not in avail:
        print(f'Backend {backend!r} not available on this machine (have: {", ".join(avail)}). '
              'Falling back to cpu.')
        backend = 'cpu'

    print(f'Using {backends.BACKENDS[backend].label} backend')

    # --trailer auditions voice + chapter detection cheaply before a full hour-long run.
    if args.trailer:
        from audiblez import doctor
        checks = doctor.run_checks(backend)
        if any(c.status == 'fail' for c in checks):
            print(doctor.format_report(checks))
            print('\033[91mPreflight failed; fix the above before auditioning a trailer.\033[0m')
            sys.exit(1)
        from audiblez.core import make_trailer
        out = os.path.join(args.output, 'trailer.wav')
        result = make_trailer(args.epub_file_path, args.voice, out, speed=args.speed, backend=backend,
                              precision=args.precision, output_folder=args.output)
        sys.exit(0 if result else 1)

    if args.precision != 'fp32':
        # Measured on Kokoro-82M / RX 7800 XT: fp16/bf16 give ~1% (noise) speedup while
        # degrading the waveform (fp16 ~0.87 similarity, bf16 ~0.06 = broken). The model
        # is small + memory-bound, so autocast does not help here. Kept opt-in for other
        # hardware/models, but warn loudly and audition the output first.
        print(f'\033[93mWarning: --precision {args.precision} changes the audio and showed no '
              'measured speedup on Kokoro-82M; audition the output (bf16 is known-broken here).\033[0m')

    cache_dir = os.path.join(args.output, '.audiblez_cache') if args.cache else None
    from audiblez.core import main
    failures = main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, backend=backend,
                    cache_dir=cache_dir, tune=args.tune, precision=args.precision)
    # Non-zero exit on a degraded run (sentences dead-lettered -> gaps), so scripts/CI notice.
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    cli_main()
