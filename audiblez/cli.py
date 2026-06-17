# -*- coding: utf-8 -*-
import argparse
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
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # --doctor checks the backend the user *requested* (not a silent CPU fallback),
    # and runs before importing the heavy TTS stack so it works on a broken install.
    if args.doctor:
        from audiblez import doctor
        requested = args.backend if args.backend is not None else ('cuda' if args.cuda else 'cpu')
        sys.exit(0 if doctor.run_doctor(backend=requested, voice=args.voice, deep=args.deep) else 1)

    if not args.epub_file_path:
        parser.error('the following arguments are required: epub_file_path (or use --doctor)')

    # --merge needs no backend/model: just stitch existing chapter wavs into an m4b.
    if args.merge:
        from audiblez.core import merge_chapters
        result = merge_chapters(args.epub_file_path, args.voice, args.output)
        sys.exit(0 if result else 1)

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
        backend = 'cpu'  # bare invocation stays on CPU (unchanged default behavior)

    if backend not in avail:
        print(f'Backend {backend!r} not available on this machine (have: {", ".join(avail)}). '
              'Falling back to cpu.')
        backend = 'cpu'

    print(f'Using {backends.BACKENDS[backend].label} backend')
    from audiblez.core import main
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, backend=backend)


if __name__ == '__main__':
    cli_main()
