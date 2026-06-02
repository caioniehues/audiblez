# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import voices, available_voices_str
from audiblez import backends


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              available_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
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
