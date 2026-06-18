# -*- coding: utf-8 -*-
"""Opt-in, per-sentence content-addressed synthesis cache (KEYSTONE slice 1).

This is the deliberately small first slice of the KEYSTONE idea: a **cache only**, with
NO resume / streaming / correctness claim. Enabled with ``audiblez --cache`` (CLI only —
the GUI does not wire the cache yet), it keys each synthesized sentence by a versioned
hash and reuses the audio on a re-run.

The key (see :func:`make_key`) captures everything that changes the waveform, the lesson
from the spike: cache version, engine, **model repo_id (which differs by quantization
between the torch and MLX backends)**, voice, speed, ``MAX_SENTENCE_LENGTH``, the spaCy
version, and the (already lexicon-transformed) sentence text. Miss anything and the cache
serves stale/wrong audio — so the key is conservative by construction.

Granularity decision: the cache unit is the **sentence**, not the batch. When caching is on,
synthesis runs per sentence so each cached unit maps 1:1 to a stored array and concatenation
seams are identical to a per-sentence run. Batching the cache-misses is a documented later
optimization (findings.md §4). Pure numpy/hashlib/json — no audio model needed to test.
"""
import os
import json
import hashlib
import numpy as np
from pathlib import Path

CACHE_VERSION = 1  # bump to invalidate every entry when the synth contract changes


def make_key(engine, repo_id, voice, speed, text, max_sentence_length, spacy_version='',
             precision='fp32'):
    """Deterministic content-addressed key for one synthesized sentence.

    ``precision`` is part of the key because fp16/bf16 autocast changes the waveform;
    omitting it would let a cache populated under one precision serve another's audio.
    """
    payload = json.dumps({
        'v': CACHE_VERSION,
        'engine': engine,
        'repo_id': repo_id,
        'voice': voice,
        'speed': round(float(speed), 4),
        'precision': precision,
        'msl': max_sentence_length,
        'spacy': spacy_version,
        'text': text,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class SynthCache:
    """Disk-backed store of per-sentence audio arrays (one ``<key>.npy`` per sentence)."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key):
        return self.dir / f'{key}.npy'

    def get(self, key):
        """Return the cached audio array for ``key`` (counting a hit), or None (a miss)."""
        path = self._path(key)
        if path.exists():
            try:
                audio = np.load(path)
                self.hits += 1
                return audio
            except (ValueError, OSError, EOFError) as e:
                # Corrupt/truncated entry (e.g. an interrupted write): drop it so it is
                # regenerated cleanly next run instead of failing np.load forever.
                print(f'Warning: discarding corrupt cache entry {path.name}: {e}')
                path.unlink(missing_ok=True)
        self.misses += 1
        return None

    def put(self, key, audio):
        """Persist an audio array under ``key`` (best-effort; cache failures never abort a run).

        Writes to a unique temp file then atomically renames, so a crash mid-write can
        never leave a half-written ``.npy`` that a later run would read as valid.
        """
        path = self._path(key)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp.npy')
        try:
            np.save(tmp, np.asarray(audio))
            os.replace(tmp, path)
        except Exception as e:  # disk full, bad array, etc. — never abort the run
            print(f'Warning: could not write cache entry: {e}')
            Path(tmp).unlink(missing_ok=True)

    def clear(self):
        """Delete every cached entry. Returns the number of files removed."""
        removed = 0
        for f in self.dir.glob('*.npy'):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def stats(self):
        total = self.hits + self.misses
        rate = (100 * self.hits // total) if total else 0
        return f'{self.hits} hits, {self.misses} misses ({rate}% hit rate)'
