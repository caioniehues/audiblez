# -*- coding: utf-8 -*-
"""Book-scoped pronunciation lexicon: fix a recurring name/acronym once, reuse everywhere.

A lexicon is a JSON sidecar (``<book>.lexicon.json``) mapping a term to its respelling,
applied to chapter text BEFORE phonemization so Kokoro speaks the respelling instead of
mangling the original. It can be auto-seeded from a book's recurring proper nouns and
acronyms, then edited by hand (or in the GUI).

Pure stdlib (json/re/hashlib) — no audio/model dependency — so it imports and tests
anywhere. :func:`fingerprint` exposes a stable hash of the active overrides for a future
content-addressed cache key (Phase 8), since the lexicon changes the synthesized waveform.
"""
import re
import json
import hashlib
from pathlib import Path
from collections import Counter

_ACRONYM_RE = re.compile(r'\b[A-Z]{2,}\b')        # NASA, USB, AI
_PROPER_RE = re.compile(r'\b[A-Z][a-z]{2,}\b')    # Capitalized words (>=3 chars)

# Common capitalized sentence-starters / function words to keep out of the seed.
_STOPWORDS = {
    'The', 'And', 'But', 'For', 'Nor', 'Yet', 'She', 'His', 'Her', 'Him', 'Its', 'Our',
    'Their', 'They', 'This', 'That', 'These', 'Those', 'Then', 'There', 'Here', 'When',
    'Where', 'What', 'Which', 'While', 'With', 'Without', 'From', 'Into', 'Over', 'After',
    'Before', 'Once', 'Now', 'Not', 'You', 'Your', 'Was', 'Were', 'Are', 'Had', 'Has',
    'Have', 'Will', 'Would', 'Could', 'Should', 'One', 'Two', 'How', 'Why', 'All',
}


def lexicon_path(file_path, output_folder='.'):
    """Sidecar path for a book's lexicon: ``<output_folder>/<book-stem>.lexicon.json``."""
    return Path(output_folder) / f'{Path(file_path).stem}.lexicon.json'


def load_lexicon(path):
    """Load a lexicon mapping from ``path``; return {} if missing or malformed."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_lexicon(path, mapping):
    """Write a lexicon mapping to ``path`` as pretty, sorted JSON. Returns the path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _active(mapping):
    """Only the entries that actually change text: non-empty respelling that differs."""
    return {k: v for k, v in mapping.items() if v and v != k}


def apply_lexicon(text, mapping):
    """Replace each lexicon term (whole word) with its respelling.

    Identity and empty entries are no-ops (a freshly seeded lexicon changes nothing until
    a value is edited). Longer terms are applied first so an overlapping shorter term does
    not pre-empt a longer one.
    """
    active = _active(mapping)
    if not active:
        return text
    for term in sorted(active, key=len, reverse=True):
        text = re.sub(rf'\b{re.escape(term)}\b', active[term].replace('\\', r'\\'), text)
    return text


def seed_terms(text, min_count=2, max_terms=200):
    """Candidate terms worth a pronunciation override: acronyms + recurring proper nouns.

    Acronyms (all-caps, >=2 chars) are always included; capitalized words are included only
    if they recur at least ``min_count`` times (filtering sentence-start noise) and are not
    common function words. Ordered by frequency, capped at ``max_terms``.
    """
    counts = Counter(_ACRONYM_RE.findall(text))
    counts.update(_PROPER_RE.findall(text))
    terms = []
    for term, count in counts.most_common():
        if term in _STOPWORDS:
            continue
        if term.isupper() or count >= min_count:
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def build_seed_lexicon(text, **kwargs):
    """Build an identity mapping {term: term} of seed candidates for the user to edit."""
    return {term: term for term in seed_terms(text, **kwargs)}


def fingerprint(mapping):
    """Stable short hash of the ACTIVE overrides, for a content-addressed cache key.

    Empty when nothing is active, so a seeded-but-unedited lexicon does not perturb the key.
    """
    active = _active(mapping)
    if not active:
        return ''
    blob = json.dumps(active, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
