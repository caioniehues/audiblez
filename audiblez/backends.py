# -*- coding: utf-8 -*-
"""Narration backend registry and capability detection.

Pure data + detection only: this module imports ``torch`` lazily (inside functions)
and never imports ``kokoro``, ``mlx_audio``, or ``audiblez.core``. That keeps it safe
to import from ``cli.py`` and ``ui.py`` without pulling in the heavy TTS stack — the
actual engine construction lives in :func:`audiblez.core.build_synthesizer`.
"""
import platform
import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    id: str                    # 'cpu' | 'cuda' | 'rocm' | 'mps' | 'mlx'
    label: str                 # human-readable label for the CLI/GUI
    engine: str                # 'torch' | 'mlx' — selects the generator in core.build_synthesizer
    torch_device: str | None   # 'cpu'|'cuda'|'mps' for torch backends; None for the mlx engine


# Declaration order is also the GUI/menu display order.
BACKENDS = {
    'cpu':  BackendInfo('cpu',  'CPU',                 'torch', 'cpu'),
    'cuda': BackendInfo('cuda', 'CUDA (NVIDIA GPU)',   'torch', 'cuda'),
    'rocm': BackendInfo('rocm', 'ROCm (AMD GPU)',      'torch', 'cuda'),  # ROCm exposes AMD GPUs via the torch.cuda API
    'mps':  BackendInfo('mps',  'MPS (Apple Silicon)', 'torch', 'mps'),
    'mlx':  BackendInfo('mlx',  'MLX (Apple Silicon)', 'mlx',   None),
}
BACKEND_IDS = tuple(BACKENDS)  # ('cpu', 'cuda', 'rocm', 'mps', 'mlx') — argparse choices


def _mlx_importable() -> bool:
    """True only on Apple Silicon with mlx-audio installed (no import side effects)."""
    return (platform.system() == 'Darwin'
            and importlib.util.find_spec('mlx_audio') is not None)


def available_backends() -> list[str]:
    """Backends actually usable on this machine, in preference/display order."""
    avail = ['cpu']  # torch is a hard dependency, so CPU is always available
    try:
        import torch
        if torch.cuda.is_available():
            # A torch wheel is built against EITHER CUDA or HIP, so at most one applies.
            if getattr(torch.version, 'hip', None) is not None:
                avail.append('rocm')   # HIP build -> AMD GPU
            elif getattr(torch.version, 'cuda', None) is not None:
                avail.append('cuda')   # CUDA build -> NVIDIA GPU
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            avail.append('mps')
    except Exception:
        pass
    if _mlx_importable():
        avail.append('mlx')
    return avail


def default_backend() -> str:
    """Best available backend: prefer an accelerator, fall back to CPU."""
    avail = available_backends()
    for pref in ('cuda', 'rocm', 'mps', 'mlx', 'cpu'):
        if pref in avail:
            return pref
    return 'cpu'


def is_gpu(backend: str) -> bool:
    """Whether a backend is GPU-accelerated (drives the chars/sec ETA estimate)."""
    return backend in ('cuda', 'rocm', 'mps', 'mlx')
