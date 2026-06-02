# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez import voices as voicelib
from audiblez import backends


def cli_main():
    epilog = ('examples:\n' +
              '  audiblez book.epub -v af_heart           # best-quality default voice\n' +
              '  audiblez book.epub -v af_warm -b mlx     # a curated "house narrator" blend\n' +
              "  audiblez book.epub -v 'af_bella:60,af_heart:40'  # your own weighted blend\n\n" +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              voicelib.available_voices_str)
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=voicelib.DEFAULT_VOICE,
                        help=(f'Narrating voice (default: {voicelib.DEFAULT_VOICE}). A voice id '
                              f'(e.g. af_heart), a preset blend ({", ".join(voicelib.PRESET_BLENDS)}), '
                              "or a custom blend like 'af_bella:60,af_heart:40'. See the list below."))
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-b', '--backend', choices=backends.BACKEND_IDS, default=None,
                        help='Narration backend: cpu, cuda (NVIDIA), rocm (AMD), mps (Apple Silicon), '
                             'mlx (Apple Silicon native). Default: cpu.')
    parser.add_argument('-c', '--cuda', default=False, action='store_true',
                        help=argparse.SUPPRESS)  # deprecated alias for --backend cuda
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Validate the voice/blend spec up front for a clean error (instead of a deep
    # traceback at synthesis time). Pure-Python, keeps the --help path torch-free.
    try:
        voicelib.parse_voice_spec(args.voice)
    except ValueError as e:
        parser.error(str(e))

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
        # A torch wheel exposes the GPU as EITHER 'cuda' (NVIDIA) or 'rocm' (AMD/HIP), never both. A user
        # who asks for one when the build provides the other still wants their GPU, so map across first.
        sibling = {'cuda': 'rocm', 'rocm': 'cuda'}.get(backend)
        if sibling in avail:
            print(f'{backend!r} requested, but this torch build exposes the GPU as {sibling!r}; using {sibling}.')
            backend = sibling
        else:
            print(f'Backend {backend!r} not available on this machine (have: {", ".join(avail)}). '
                  'Falling back to cpu.')
            backend = 'cpu'

    print(f'Using {backends.BACKENDS[backend].label} backend')
    from audiblez.core import main
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, backend=backend)


if __name__ == '__main__':
    cli_main()
