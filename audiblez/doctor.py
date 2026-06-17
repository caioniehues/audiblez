# -*- coding: utf-8 -*-
"""Preflight checks — fail fast and LOUD before an hour of synthesis.

Pure and import-light by design: this module imports only stdlib plus
:mod:`audiblez.backends` (which itself imports nothing heavy). It never imports
``torch``/``kokoro``/``spacy``/``phonemizer`` at module scope, so ``audiblez
--doctor`` stays fast and works even when the heavy TTS stack is half-installed —
which is exactly the situation it exists to diagnose. The optional ``--deep``
check loads the real model and is the only path that imports :mod:`audiblez.core`.
"""
import os
import platform
import shutil
import subprocess
import importlib.util
from glob import glob
from pathlib import Path
from dataclasses import dataclass

from audiblez import backends

# ANSI colours, matching the red used elsewhere in core.py.
_COLORS = {'ok': '\033[92m', 'warn': '\033[93m', 'fail': '\033[91m'}
_MARKS = {'ok': '[ OK ]', 'warn': '[WARN]', 'fail': '[FAIL]'}
_RESET = '\033[0m'

SPACY_MODEL = 'xx_ent_wiki_sm'


@dataclass
class CheckResult:
    name: str
    status: str   # 'ok' | 'warn' | 'fail'
    detail: str


def find_espeak_library() -> str:
    """Resolve the espeak-ng shared-library path, or raise RuntimeError.

    Pure path resolution (``ESPEAK_LIBRARY`` env override / platform glob /
    Homebrew Cellar). Does NOT import phonemizer or register anything — it is both
    the preflight espeak check and the resolver used by
    :func:`audiblez.core.set_espeak_library`. Raises with an actionable, OS-specific
    install hint instead of the old swallow-and-continue.
    """
    override = os.environ.get('ESPEAK_LIBRARY')
    if override:
        if not Path(override).exists():
            raise RuntimeError(f'ESPEAK_LIBRARY={override!r} is set but the file does not exist')
        return override

    system = platform.system()
    if system == 'Darwin':
        try:
            cellar = Path(subprocess.check_output(['brew', '--cellar'], text=True).strip())
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Cannot locate the Homebrew Cellar (is 'brew' installed and on PATH?). "
                'Install espeak-ng with: brew install espeak-ng') from e
        lib = next(iter(glob(str(cellar / 'espeak-ng' / '*' / 'lib' / '*.dylib'))), None)
        if not lib:
            raise RuntimeError('No espeak-ng library found. Install it with: brew install espeak-ng')
        return lib
    if system == 'Linux':
        libs = (glob('/usr/lib/*/libespeak-ng*')
                + glob('/usr/lib/libespeak-ng*')
                + glob('/usr/local/lib/libespeak-ng*'))
        if not libs:
            raise RuntimeError('No espeak-ng library found. Install it with: sudo apt install espeak-ng')
        return libs[0]
    if system == 'Windows':
        lib = next(iter(glob('C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll')), None)
        if not lib:
            raise RuntimeError(
                'No espeak-ng library found. Install eSpeak NG from '
                'https://github.com/espeak-ng/espeak-ng/releases')
        return lib
    raise RuntimeError(
        f'Unsupported OS {system!r}; set ESPEAK_LIBRARY to the espeak-ng library path manually')


def check_ffmpeg() -> CheckResult:
    path = shutil.which('ffmpeg')
    return CheckResult('ffmpeg', 'ok' if path else 'fail',
                       path or 'not found on PATH — install ffmpeg to write .m4b/.wav output')


def check_ffprobe() -> CheckResult:
    # ffprobe only sharpens chapter-marker timing; its absence degrades, never fails.
    path = shutil.which('ffprobe')
    return CheckResult('ffprobe', 'ok' if path else 'warn',
                       path or 'not found — chapter markers will fall back to file size')


def check_espeak() -> CheckResult:
    try:
        return CheckResult('espeak-ng', 'ok', find_espeak_library())
    except RuntimeError as e:
        return CheckResult('espeak-ng', 'fail', str(e))


def check_kokoro() -> CheckResult:
    if importlib.util.find_spec('kokoro') is None:
        return CheckResult('kokoro', 'fail', 'not installed (pip install kokoro==0.9.4)')
    return CheckResult('kokoro', 'ok', 'installed')


def check_spacy_model(model: str = SPACY_MODEL) -> CheckResult:
    if importlib.util.find_spec('spacy') is None:
        return CheckResult('spaCy', 'fail', 'not installed (pip install spacy)')
    if importlib.util.find_spec(model) is None:
        return CheckResult(f'spaCy model {model}', 'warn',
                           f'not installed — audiblez will auto-download it on first run')
    return CheckResult(f'spaCy model {model}', 'ok', 'installed')


def check_backend(backend: str) -> CheckResult:
    if backend not in backends.BACKENDS:
        return CheckResult(f'backend {backend}', 'fail',
                           f'unknown; choose from {", ".join(backends.BACKEND_IDS)}')
    info = backends.BACKENDS[backend]
    avail = backends.available_backends()
    if info.engine == 'torch':
        if importlib.util.find_spec('torch') is None:
            return CheckResult(f'backend {backend}', 'fail', 'torch not installed (pip install torch)')
        if backend not in avail:
            return CheckResult(f'backend {backend}', 'fail',
                               f'device not available here (usable: {", ".join(avail)})')
        return CheckResult(f'backend {backend}', 'ok', info.label)
    # mlx engine
    if backend not in avail:
        return CheckResult(f'backend {backend}', 'fail',
                           'requires Apple Silicon + mlx-audio (pip install "audiblez[mlx]")')
    return CheckResult(f'backend {backend}', 'ok', info.label)


def run_checks(backend: str = 'cpu') -> list[CheckResult]:
    """Fast preflight: tools + deps + the selected backend. No model load."""
    return [check_ffmpeg(), check_ffprobe(), check_espeak(),
            check_kokoro(), check_spacy_model(), check_backend(backend)]


def deep_check(backend: str = 'cpu', voice: str = 'af_sky') -> CheckResult:
    """Slow check: actually build the synthesizer and synth one word.

    Loads the full model (seconds), so it is opt-in (``--doctor --deep``) and the
    only check that imports :mod:`audiblez.core`.
    """
    try:
        import audiblez.core as core
        core.set_espeak_library()
        synth = core.build_synthesizer(voice, backend)
        out = synth('Hello.', 1.0)
        produced = bool(out) and sum(len(seg) for seg in out) > 0
        return CheckResult('deep: synth one word', 'ok' if produced else 'fail',
                           'produced audio' if produced else 'synth returned no audio')
    except Exception as e:  # any failure in the heavy path is a diagnostic, not a crash
        return CheckResult('deep: synth one word', 'fail', f'{type(e).__name__}: {e}')


def format_report(results: list[CheckResult], color: bool = True) -> str:
    lines = []
    for r in results:
        line = f'{_MARKS[r.status]} {r.name}: {r.detail}'
        if color:
            line = f'{_COLORS[r.status]}{line}{_RESET}'
        lines.append(line)
    return '\n'.join(lines)


def run_doctor(backend: str = 'cpu', voice: str = 'af_sky', deep: bool = False,
               color: bool = True, out=print) -> bool:
    """Run the preflight, print a red/green report, and return True iff nothing failed."""
    results = run_checks(backend)
    if deep:
        results.append(deep_check(backend, voice))
    out(format_report(results, color=color))
    ok = not any(r.status == 'fail' for r in results)
    summary = 'All required checks passed.' if ok else 'Some required checks FAILED.'
    out('')
    out(f'{_COLORS["ok" if ok else "fail"]}{summary}{_RESET}' if color else summary)
    return ok
