# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path

from audiblez import voices as voicelib
from audiblez import backends


def parse_chapter_spec(spec: str) -> list[int]:
    """Parse a --chapters spec like '1,3,5' or '1-4,7' into a sorted, unique list of 1-based ints.

    Pure helper (no argparse dependency) so it is easy to test. Raises ValueError on any
    malformed token (non-integer, empty, inverted range, or non-positive index); the caller
    turns that into a clean parser.error().
    """
    result: set[int] = set()
    for token in spec.split(','):
        token = token.strip()
        if not token:
            raise ValueError(f'empty chapter token in {spec!r}')
        if '-' in token:
            start_str, _, end_str = token.partition('-')
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                raise ValueError(f'invalid chapter range {token!r}; expected integers like 1-4')
            if start > end:
                raise ValueError(f'inverted chapter range {token!r}; start must be <= end')
            if start < 1:
                raise ValueError(f'chapter numbers are 1-based; {start} is out of range in {token!r}')
            result.update(range(start, end + 1))
        else:
            try:
                n = int(token)
            except ValueError:
                raise ValueError(f'invalid chapter number {token!r}; expected an integer')
            if n < 1:
                raise ValueError(f'chapter numbers are 1-based; {n} is out of range')
            result.add(n)
    return sorted(result)


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
    parser.add_argument('--chapters', default=None, metavar='SPEC',
                        help="Select chapters by 1-based index without the interactive picker, "
                             "e.g. '1,3,5' or '1-4,7'. Headless alternative to --pick.")
    parser.add_argument('--chapter-text-dir', default=None, metavar='DIR',
                        help='Directory of per-chapter text overrides to read instead of the extracted epub text.')
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

    # Headless chapter selection. Parsed up front so a bad spec fails fast and cleanly.
    selected_chapters = None
    if args.chapters is not None:
        try:
            selected_chapters = parse_chapter_spec(args.chapters)
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
    chapter_text_dir = Path(args.chapter_text_dir) if args.chapter_text_dir is not None else None
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, backend=backend,
         selected_chapters=selected_chapters, chapter_text_dir=chapter_text_dir)


if __name__ == '__main__':
    cli_main()
