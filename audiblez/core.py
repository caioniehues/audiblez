#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import spacy
import ebooklib
import soundfile
import numpy as np
import json
import time
import shutil
import subprocess
import platform
import re
from glob import glob
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pick import pick

from audiblez import backends

sample_rate = 24000
_nlp = None  # cached spaCy pipeline (loaded once, reused across chapters/previews)

MAX_SENTENCE_LENGTH = 400  # chars; Kokoro truncates long non-English sentences, so we split them
GPU_CHARS_PER_SEC = 500    # rough synthesis throughput, used only for the ETA display
CPU_CHARS_PER_SEC = 50

# Truncation floor for validate-before-skip: no real narration packs more than this
# many characters into a second of audio, so anything shorter than text/this is too
# short to be a complete chapter. Deliberately generous to avoid deleting valid wavs.
VALIDATION_MAX_CHARS_PER_SEC = 30

EWMA_ALPHA = 0.3       # weight of the newest measurement in the rolling chars/sec ETA
HEARTBEAT_EVERY = 10   # emit a progress heartbeat line every N synthesized sentences

# Batch multiple sentences into one synth() call (Kokoro re-splits on \n\n\n internally),
# trading per-call Python/model overhead for throughput. Set to a tiny value to disable
# batching and fall back to one sentence per call.
BATCH_MAX_CHARS = 1000

SYNTH_RETRIES = 2                     # extra attempts after the first before giving up on a unit
FALLBACK_SILENCE_CHARS_PER_SEC = 15   # gap length (proportional to text) for a dead-lettered sentence

TRAILER_SENTENCES_PER_CHAPTER = 2     # opening sentences sampled per chapter in --trailer
TRAILER_GAP_SECONDS = 1.0             # silent gap between trailer segments


def to_numpy(audio):
    """Normalise a kokoro audio segment to a 1-D numpy array.

    Depending on version/device kokoro may yield a torch.Tensor (possibly on GPU)
    or a numpy array; this handles both so np.concatenate / soundfile.write work.
    """
    if hasattr(audio, 'detach'):  # torch.Tensor
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def load_spacy():
    """Load (once) and cache the multilingual spaCy model used for sentence splitting.

    xx_ent_wiki_sm has no sentence boundaries on its own, so a 'sentencizer' is
    added the first time the model is loaded (guarded in case a future model ships
    one). Caching at module level avoids reloading the model from disk for every
    chapter and preview, which was a major performance drain.
    """
    global _nlp
    if _nlp is None:
        if not spacy.util.is_package("xx_ent_wiki_sm"):
            print("Downloading Spacy model xx_ent_wiki_sm...")
            spacy.cli.download("xx_ent_wiki_sm")
        _nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def set_espeak_library():
    """Locate the espeak-ng library and register it with phonemizer.

    Fails LOUD: the path resolution (delegated to :func:`audiblez.doctor.find_espeak_library`)
    raises a ``RuntimeError`` with an actionable, OS-specific install hint instead of the
    old swallow-and-continue, which used to let a run proceed for an hour and emit nothing.
    Run ``audiblez --doctor`` to check this ahead of a long synthesis.
    """
    from audiblez.doctor import find_espeak_library
    library = find_espeak_library()
    print('Using espeak library:', library)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(library)


def extract_book_metadata(book):
    """Return (title, creator) from an ebooklib book's Dublin Core metadata."""
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''
    return title, creator


def main(file_path: str, voice: str, pick_manually: bool, speed: float, output_folder: str = '.',
         max_chapters: int | None = None, max_sentences: int | None = None,
         selected_chapters: list | None = None, backend: str = 'cpu', post_event=None) -> None:
    if post_event: post_event('CORE_STARTED')
    # Fast preflight: abort in seconds with an actionable message rather than dying
    # 40 minutes in on a missing dep. See `audiblez --doctor` for the same checks.
    from audiblez import doctor
    checks = doctor.run_checks(backend)
    if any(c.status == 'fail' for c in checks):
        print(doctor.format_report(checks))
        failed = '; '.join(f'{c.name}: {c.detail}' for c in checks if c.status == 'fail')
        raise RuntimeError(f'Preflight failed — fix these before synthesis: {failed}')
    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name

    book = epub.read_epub(file_path)
    title, creator = extract_book_metadata(book)

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    document_chapters = find_document_chapters_and_extract_texts(book)

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)
    texts = [c.extracted_text for c in selected_chapters]

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        sentences_done=0,
        # Flat constant is only the initial prior; the real rate is measured per
        # sentence and folded into an EWMA as synthesis proceeds (see _update_eta).
        chars_per_sec=GPU_CHARS_PER_SEC if backends.is_gpu(backend) else CPU_CHARS_PER_SEC)
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    set_espeak_library()
    synth = build_synthesizer(voice, backend)

    chapter_wav_files = []
    intro_added = False
    for i, chapter in enumerate(selected_chapters, start=1):
        if max_chapters and i > max_chapters: break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        chapter_wav_path = Path(output_folder) / f'{Path(filename).stem}_chapter_{i}_{voice}_{xhtml_file_name}.wav'
        chapter_wav_files.append(chapter_wav_path)
        if Path(chapter_wav_path).exists():
            # Only skip if the existing wav is actually usable — a truncated/zero-byte
            # file from a prior crash must be regenerated, not reused into the m4b.
            expected_len = None if max_sentences else len(text)
            if is_valid_chapter_wav(chapter_wav_path, expected_len):
                print(f'File for chapter {i} already exists. Skipping')
                stats.processed_chars += len(text)
                if post_event:
                    post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
                continue
            print(f'Existing file for chapter {i} is invalid/truncated; regenerating.')
            Path(chapter_wav_path).unlink(missing_ok=True)
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)
            continue
        if not intro_added:
            # Prepend the book intro to the first chapter actually synthesized
            # (not necessarily i == 1, which may have been skipped or empty).
            text = f'{title} – {creator}.\n\n' + text
            intro_added = True
        start_time = time.time()
        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)
        # Fresh dead-letter per (re)generated chapter; failed sentences are appended here.
        dead_letter_path = Path(chapter_wav_path).with_suffix('.failed.jsonl')
        Path(dead_letter_path).unlink(missing_ok=True)
        audio_segments = gen_audio_segments(
            synth, text, voice, speed, stats, post_event=post_event, max_sentences=max_sentences,
            chapter_label=f'chapter {i}', dead_letter_path=dead_letter_path)
        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            soundfile.write(chapter_wav_path, final_audio, sample_rate)
            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds
            print('Chapter written to', chapter_wav_path)
            if post_event: post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
        else:
            print(f'Warning: No audio generated for chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)

    if has_ffmpeg:
        # Assemble only valid chapter audio, so one corrupt/short wav can't break the
        # whole m4b; completed chapters always persist as wavs and can also be assembled
        # later with `audiblez --merge` if the run is interrupted before this point.
        valid_wavs = [w for w in chapter_wav_files if is_valid_chapter_wav(w, None)]
        if valid_wavs:
            create_m4b(valid_wavs, filename, cover_image, output_folder, title=title, creator=creator)
            if post_event: post_event('CORE_FINISHED')
        else:
            print('No valid chapter audio to assemble into an m4b.')


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    # https://idpf.org/forum/topic-715
    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    selected_ids = {id(c) for c in chapters}
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if id(c) in selected_ids else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))

def split_long_sentence(text, max_length=MAX_SENTENCE_LENGTH):
    """Split a long sentence near max_length chars, breaking at the last whitespace before it."""
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()
    if text:
        parts.append(text)
    return parts


def build_synthesizer(voice: str, backend: str = 'cpu'):
    """Return ``synth(text, speed) -> list[np.ndarray]`` (float32 @ 24000 Hz).

    Torch backends (cpu/cuda/rocm/mps) set the process-global default device and build
    a Kokoro KPipeline; the mlx backend loads the Apple-Silicon-native Kokoro model.
    Both engines emit identical-format audio, so everything downstream is engine-agnostic.
    """
    if backend not in backends.BACKENDS:
        raise ValueError(f'Unknown backend {backend!r}; choose from {backends.BACKEND_IDS}')
    info = backends.BACKENDS[backend]
    lang_code = voice[0]
    if info.engine == 'mlx':
        if not backends._mlx_importable():
            raise RuntimeError(
                "Backend 'mlx' requires mlx-audio on Apple Silicon. "
                'Install it with: pip install "audiblez[mlx]"')
        return _build_mlx_synth(voice, lang_code)
    # torch path (cpu/cuda/rocm/mps): KPipeline's device= does a proper model .to(device),
    # which (unlike a global torch.set_default_device) actually works on MPS and avoids
    # mutating process-wide torch state.
    pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M', device=info.torch_device)

    def synth(text, speed):
        return [to_numpy(audio)
                for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r'\n\n\n')]
    return synth


def _build_mlx_synth(voice: str, lang_code: str):
    """Engine closure for the Apple-Silicon-native MLX backend (optional dependency).

    mlx-audio is imported here, never at module scope, so importing audiblez.core does
    not require it on non-Apple platforms.
    """
    from mlx_audio.tts.utils import load_model  # optional dep; only when mlx is selected
    model = load_model('mlx-community/Kokoro-82M-bf16')

    def synth(text, speed):
        return [np.asarray(seg.audio).reshape(-1)
                for seg in model.generate(text, voice=voice, speed=speed,
                                          lang_code=lang_code, split_pattern=r'\n\n\n')]
    return synth


def ewma(prev, sample, alpha=EWMA_ALPHA):
    """Exponentially-weighted moving average of a throughput sample.

    ``prev`` is the previous estimate (e.g. the flat GPU/CPU_CHARS_PER_SEC prior on the
    first call); ``None`` seeds the average with the first real sample.
    """
    if prev is None:
        return sample
    return alpha * sample + (1 - alpha) * prev


def _update_eta(stats, measured_chars, elapsed):
    """Fold one measurement into stats: rolling chars/sec, progress %, and ETA string."""
    stats.processed_chars += measured_chars
    if elapsed and elapsed > 0:
        stats.chars_per_sec = ewma(getattr(stats, 'chars_per_sec', None), measured_chars / elapsed)
    stats.progress = min(100, stats.processed_chars * 100 // stats.total_chars) if stats.total_chars else 100
    remaining_chars = max(0, stats.total_chars - stats.processed_chars)  # intro chars can overshoot
    stats.eta = strfdelta(remaining_chars / (stats.chars_per_sec or 1))


def pack_sentences(sentences, batch_max_chars=BATCH_MAX_CHARS):
    """Group consecutive sentences into batches of at most ``batch_max_chars`` chars.

    A batch is joined with ``\\n\\n\\n`` and handed to synth in one call. A single
    sentence longer than the cap becomes its own (oversized) batch rather than being
    dropped or merged — preserving the per-sentence fallback for long sentences.
    Order is always preserved, so chapter audio concatenates identically.
    """
    batches, current, current_len = [], [], 0
    for s in sentences:
        if current and current_len + len(s) > batch_max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(s)
        current_len += len(s)
    if current:
        batches.append(current)
    return batches


def _silence_for(text):
    """A silent placeholder segment, length roughly proportional to the failed text."""
    seconds = max(0.3, len(text) / FALLBACK_SILENCE_CHARS_PER_SEC)
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _retry(fn, retries):
    """Call ``fn`` up to ``retries + 1`` times (at least once); return it or re-raise the last error."""
    last = RuntimeError('no attempts made')
    for _ in range(max(1, retries + 1)):
        try:
            return fn()
        except Exception as e:
            last = e
    raise last


def _record_dead_letter(path, chapter_label, text, error):
    """Append one failed sentence to a chapter's ``.failed.jsonl`` dead-letter file."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'chapter': chapter_label, 'text': text, 'error': error},
                               ensure_ascii=False) + '\n')
    except OSError as e:
        print(f'Warning: could not write dead-letter entry to {path}: {e}')


def _synth_batch(synth, batch, speed, retries=SYNTH_RETRIES, chapter_label=None, dead_letter_path=None):
    """Synthesize a batch of sentences, never raising for a single bad sentence.

    Tries the whole batch in one call (with bounded retries). On persistent failure it
    falls back to per-sentence synthesis; any sentence that still fails after its retries
    is replaced with silence and recorded to a dead-letter file, so one bad sentence (or
    a failed batched call) degrades to a gap instead of nuking the entire chapter.
    """
    try:
        return _retry(lambda: synth('\n\n\n'.join(batch), speed), retries)
    except Exception as e:
        print(f'\033[91mBatch synth failed ({e}); retrying sentence-by-sentence.\033[0m')
    segments = []
    for s in batch:
        try:
            segments.extend(_retry(lambda s=s: synth(s, speed), retries))
        except Exception as e:
            print(f'\033[91mSentence failed after {retries + 1} attempts; inserting silence.\033[0m')
            segments.append(_silence_for(s))
            if dead_letter_path:
                _record_dead_letter(dead_letter_path, chapter_label, s, str(e))
    return segments


def gen_audio_segments(synth, text, voice, speed, stats=None, max_sentences=None, post_event=None,
                       chapter_label=None, batch_max_chars=BATCH_MAX_CHARS,
                       synth_retries=SYNTH_RETRIES, dead_letter_path=None):
    nlp = load_spacy()
    audio_segments = []
    doc = nlp(text)
    lang_code = voice[0]

    if lang_code in 'ab':
        sentences = [s.text for s in doc.sents]
    else:
        # For non-english languages, Kokoro truncates long sentences, so we split them manually
        sentences = []
        for sent in list(doc.sents):
            if len(sent.text) > MAX_SENTENCE_LENGTH:
                print(f'Warning: Sentence too long ({len(sent.text)} chars), splitting into smaller sentences.')
                sents = split_long_sentence(sent.text, MAX_SENTENCE_LENGTH)
                sentences.extend(sents)
            else:
                sentences.append(sent.text)

    if max_sentences:
        sentences = sentences[:max_sentences]

    for batch in pack_sentences(sentences, batch_max_chars):
        t0 = time.time()
        # Kokoro re-splits on \n\n\n, yielding one audio segment per sentence, in order.
        # Bounded retries + per-sentence fallback so a bad sentence becomes a gap, not a crash.
        audio_segments.extend(_synth_batch(synth, batch, speed, retries=synth_retries,
                                           chapter_label=chapter_label,
                                           dead_letter_path=dead_letter_path))
        elapsed = time.time() - t0
        if stats:
            before = getattr(stats, 'sentences_done', 0)
            _update_eta(stats, sum(len(s) for s in batch), elapsed)
            stats.sentences_done = before + len(batch)
            if post_event: post_event('CORE_PROGRESS', stats=stats)
            # Heartbeat when this batch crossed a HEARTBEAT_EVERY boundary.
            if stats.sentences_done // HEARTBEAT_EVERY > before // HEARTBEAT_EVERY:
                where = f' [{chapter_label}]' if chapter_label else ''
                print(f'♥ heartbeat{where}: {stats.sentences_done} sentences, '
                      f'{stats.chars_per_sec:.0f} chars/sec (measured), ETA {stats.eta}')
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False, backend='cpu'):
    load_spacy()
    synth = build_synthesizer(voice, backend)
    audio_segments = gen_audio_segments(synth, text, voice=voice, speed=speed)
    if not audio_segments:
        print('Warning: no audio generated for the given text.')
        return
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


def make_trailer(file_path, voice, output_file='trailer.wav', speed=1.0, backend='cpu',
                 sentences_per_chapter=TRAILER_SENTENCES_PER_CHAPTER, selected_chapters=None,
                 max_chapters=None):
    """Render a short audio sampler of a book: the opening sentences of each chapter.

    For each detected chapter, speaks a "Chapter N" label then its first
    ``sentences_per_chapter`` sentences, separated by short silent gaps. Lets a user
    hear in a couple of minutes whether chapter detection, voice, and pronunciation are
    right BEFORE committing to a full hour-long synthesis — and audibly reveals a
    misdetected chapter. Modeled on :func:`gen_text`; returns the output path (or None).
    """
    load_spacy()
    book = epub.read_epub(file_path)
    document_chapters = find_document_chapters_and_extract_texts(book)
    chapters = selected_chapters or find_good_chapters(document_chapters)
    synth = build_synthesizer(voice, backend)
    gap = np.zeros(int(TRAILER_GAP_SECONDS * sample_rate), dtype=np.float32)

    pieces, sampled = [], 0
    for n, chapter in enumerate(chapters, start=1):
        if max_chapters and sampled >= max_chapters:
            break
        text = chapter.extracted_text.strip()
        if len(text) < 10:
            continue
        pieces.extend(gen_audio_segments(synth, f'Chapter {n}.', voice=voice, speed=speed))
        pieces.append(gap)
        pieces.extend(gen_audio_segments(synth, text, voice=voice, speed=speed,
                                         max_sentences=sentences_per_chapter))
        pieces.append(gap)
        sampled += 1

    if not pieces:
        print('No chapters to put in the trailer.')
        return None
    soundfile.write(output_file, np.concatenate(pieces), sample_rate)
    print(f'Trailer written to {output_file} ({sampled} chapter(s) sampled).')
    return output_file


def find_document_chapters_and_extract_texts(book):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each chapter with extracted_text."""
    document_chapters = []
    for chapter in book.get_items():
        if chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter.extracted_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for text in [c.text.strip() for c in soup.find_all(html_content_tags) if c.text]:
            if not text.endswith('.'):
                text += '.'
            chapter.extracted_text += text + '\n'
        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i  # this is used in the UI to identify chapters
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name.lower()
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10]
    return chapters


def pick_chapters(chapters):
    # Display the document name, the length and first 50 characters of the text
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_ids = {id(c) for c in selected_chapters_out_of_order}
    selected_chapters = [c for c in chapters if id(c) in selected_ids]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def _escape_concat_path(path):
    """Escape a path for ffmpeg's concat demuxer list file (single-quote syntax).

    ffmpeg's concat demuxer wraps each entry as ``file '<path>'``; a literal single
    quote inside the path must be written as ``'\\''`` (close quote, escaped quote,
    reopen quote), otherwise paths containing apostrophes break the parse.
    """
    return str(path).replace("'", "'\\''")


def _escape_ffmetadata(value):
    """Escape a value for the FFMETADATA1 format.

    The format is INI-like: ``=``, ``;``, ``#`` and ``\\`` are special and newlines
    delimit fields. Escaping these prevents untrusted epub title/author metadata from
    injecting extra fields.
    """
    text = str(value)
    for ch in ('\\', '=', ';', '#'):
        text = text.replace(ch, '\\' + ch)
    return text.replace('\r', ' ').replace('\n', ' ')


def probe_duration(file_name):
    """Return an audio file's duration in seconds, or None if it can't be probed."""
    args = ['ffprobe', '-i', str(file_name), '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=True)
        return float(proc.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        # FileNotFoundError = ffprobe not on PATH; callers must degrade gracefully.
        print(f'Warning: could not probe duration of {file_name}: {e}')
        return None


def is_valid_chapter_wav(path, expected_text_len, min_chars_per_sec=VALIDATION_MAX_CHARS_PER_SEC):
    """Whether an existing chapter wav is safe to reuse instead of regenerating.

    Guards the resume-by-skip path against truncated or zero-byte wavs left behind by
    a prior crash, which would otherwise be silently piped into the final m4b. Checks,
    in order of preference and degrading gracefully:
      1. exists and is non-empty;
      2. duration is plausible for the text — via ffprobe, falling back to soundfile's
         header (when ffprobe is absent), then to a raw file-size floor;
    ``expected_text_len`` of None means the caller capped output (``max_sentences``),
    so the length is not comparable and only readability is required.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False

    duration = probe_duration(p)
    if duration is None:  # ffprobe missing/failed -> read the wav header directly
        try:
            duration = soundfile.info(str(p)).duration
        except Exception:
            duration = None

    if duration is None:
        # Can't measure duration at all: accept any file too big to be a bare header.
        return p.stat().st_size > 1024
    if duration <= 0:
        return False
    if expected_text_len is None:
        return True  # length not comparable; readable + non-empty is enough
    expected_min_sec = expected_text_len / min_chars_per_sec
    return duration >= expected_min_sec * 0.5


def create_index_file(title, creator, chapter_files, output_folder):
    """Write an FFMETADATA1 file with sequential, gap-free chapter markers.

    Numbering runs 1..N over the chapters that actually made it into the audiobook,
    so skipped/empty chapters no longer produce 'Chapter 0' or off-by-one labels.
    Returns the path to the written file.
    """
    chapters_txt_path = Path(output_folder) / "chapters.txt"
    with open(chapters_txt_path, "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={_escape_ffmetadata(title)}\n"
                f"artist={_escape_ffmetadata(creator)}\n\n")
        start = 0
        for i, c in enumerate(chapter_files, start=1):
            duration = probe_duration(c) or 0.0
            end = start + int(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            start = end
    return chapters_txt_path


def create_m4b(chapter_files, filename, cover_image, output_folder, title='', creator=''):
    """Create the final .m4b in a SINGLE ffmpeg pass: concat + chapter metadata + cover.

    Fixes several issues with the previous two-pass implementation:
      - Uses the native 'aac' encoder (present in every ffmpeg build) instead of
        'libfdk_aac' (absent from standard apt/brew builds), which previously made
        m4b creation fail for typical users.
      - Encodes only once (the old code transcoded WAV->192k AAC->64k AAC).
      - Escapes concat paths so chapter filenames with apostrophes work.
      - Raises RuntimeError on ffmpeg failure so callers never report a false success
        over a missing/corrupt output file.
      - Cleans up temp files via try/finally even when ffmpeg fails.
    """
    output_folder = Path(output_folder)
    final_filename = output_folder / (Path(filename).stem + '.m4b')
    wav_list_txt = output_folder / (Path(filename).stem + '_wav_list.txt')
    cover_file_path = None
    print('Creating M4B file...')
    try:
        with open(wav_list_txt, 'w') as f:
            for wav_file in chapter_files:
                f.write(f"file '{_escape_concat_path(wav_file)}'\n")

        chapters_txt_path = create_index_file(title, creator, chapter_files, output_folder)

        ffmpeg_args = ['ffmpeg', '-y',
                       '-f', 'concat', '-safe', '0', '-i', str(wav_list_txt),
                       '-i', str(chapters_txt_path)]
        if cover_image:
            cover_file_path = output_folder / 'cover'
            with open(cover_file_path, 'wb') as f:
                f.write(cover_image)
            ffmpeg_args += ['-i', str(cover_file_path)]

        ffmpeg_args += ['-map', '0:a', '-map_metadata', '1']
        if cover_file_path:
            ffmpeg_args += ['-map', '2:v', '-disposition:v', 'attached_pic', '-c:v', 'copy']
        ffmpeg_args += ['-c:a', 'aac', '-b:a', '64k', '-f', 'mp4', str(final_filename)]

        proc = subprocess.run(ffmpeg_args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f'ffmpeg failed to create the m4b (exit {proc.returncode}).\n'
                f'{(proc.stderr or "")[-2000:]}')
        print(f'{final_filename} created. Enjoy your audiobook.')
        print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')
        return final_filename
    finally:
        for tmp in (wav_list_txt, cover_file_path):
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)


def find_chapter_wavs(file_path, voice, output_folder='.'):
    """Return existing chapter wavs for a book/voice, ordered by chapter index.

    Matches the naming scheme used by :func:`main`
    (``<stem>_chapter_<i>_<voice>_<xhtml>.wav``) and sorts on the integer ``<i>`` so
    the merge order is the synthesis order, not lexicographic (chapter_10 after 9).
    """
    stem = Path(file_path).stem
    matches = glob(str(Path(output_folder) / f'{stem}_chapter_*_{voice}_*.wav'))

    def chapter_index(path):
        m = re.search(rf'{re.escape(stem)}_chapter_(\d+)_', Path(path).name)
        return int(m.group(1)) if m else 0

    return [Path(p) for p in sorted(matches, key=chapter_index)]


def merge_chapters(file_path, voice, output_folder='.'):
    """Assemble already-synthesized chapter wavs into a playable m4b.

    The recovery path for an interrupted run: completed chapters persist as wavs, so
    even if synthesis (or the final mux) died partway, ``audiblez --merge`` stitches the
    valid ones into an m4b. Pulls title/author/cover from the epub for metadata. Returns
    the m4b path, or None if there is nothing valid to merge / ffmpeg is missing.
    """
    if not shutil.which('ffmpeg'):
        print('\033[91mffmpeg not found; cannot merge chapters into an m4b.\033[0m')
        return None
    book = epub.read_epub(file_path)
    title, creator = extract_book_metadata(book)
    cover = find_cover(book)
    cover_image = cover.get_content() if cover else b''
    wavs = [w for w in find_chapter_wavs(file_path, voice, output_folder)
            if is_valid_chapter_wav(w, None)]
    if not wavs:
        print('No completed/valid chapter wavs found to merge.')
        return None
    print(f'Merging {len(wavs)} completed chapter(s) into an m4b...')
    return create_m4b(wavs, Path(file_path).name, cover_image, output_folder,
                      title=title, creator=creator)
