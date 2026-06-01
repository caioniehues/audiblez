#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import os
import traceback
from glob import glob

import torch.cuda
import spacy
import ebooklib
import soundfile
import numpy as np
import time
import shutil
import subprocess
import platform
import re
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pick import pick

sample_rate = 24000
_nlp = None  # cached spaCy pipeline (loaded once, reused across chapters/previews)


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
    """Find the espeak library path"""
    try:

        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            library = 'C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll'
        else:
            print('Unsupported OS, please set the espeak library path manually')
            return
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print("Error finding espeak-ng library:")
        print("Probably you haven't installed espeak-ng.")
        print("On Mac: brew install espeak-ng")
        print("On Linux: sudo apt install espeak-ng")


def main(file_path, voice, pick_manually, speed, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None):
    if post_event: post_event('CORE_STARTED')
    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name

    extension = '.epub'
    book = epub.read_epub(file_path)
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

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
        chars_per_sec=500 if torch.cuda.is_available() else 50)
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    set_espeak_library()
    pipeline = KPipeline(lang_code=voice[0], repo_id='hexgrad/Kokoro-82M')  # a=american, b=british, ...

    chapter_wav_files = []
    intro_added = False
    for i, chapter in enumerate(selected_chapters, start=1):
        if max_chapters and i > max_chapters: break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        chapter_wav_path = Path(output_folder) / filename.replace(extension, f'_chapter_{i}_{voice}_{xhtml_file_name}.wav')
        chapter_wav_files.append(chapter_wav_path)
        if Path(chapter_wav_path).exists():
            print(f'File for chapter {i} already exists. Skipping')
            stats.processed_chars += len(text)
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
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
        audio_segments = gen_audio_segments(
            pipeline, text, voice, speed, stats, post_event=post_event, max_sentences=max_sentences)
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
        create_m4b(chapter_wav_files, filename, cover_image, output_folder, title=title, creator=creator)
        if post_event: post_event('CORE_FINISHED')


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
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))

def split_long_sentence(text, max_length=400):
    """Split a long sentence around the 500 chars, picking the first whitespace after the 500th character. """
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


def gen_audio_segments(pipeline, text, voice, speed, stats=None, max_sentences=None, post_event=None):
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
            if len(sent.text) > 400:
                print(f'Warning: Sentence too long ({len(sent.text)} chars), splitting into smaller sentences.')
                sents = split_long_sentence(sent.text, 400)
                sentences.extend(sents)
            else:
                sentences.append(sent.text)

    for i, sent_text in enumerate(sentences):
        if max_sentences and i >= max_sentences: break
        for gs, ps, audio in pipeline(sent_text, voice=voice, speed=speed, split_pattern=r'\n\n\n'):
            audio_segments.append(to_numpy(audio))
        if stats:
            stats.processed_chars += len(sent_text)
            stats.progress = stats.processed_chars * 100 // stats.total_chars
            stats.eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
            if post_event: post_event('CORE_PROGRESS', stats=stats)
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False):
    lang_code = voice[:1]
    pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
    load_spacy()
    audio_segments = gen_audio_segments(pipeline, text, voice=voice, speed=speed)
    if not audio_segments:
        print('Warning: no audio generated for the given text.')
        return
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


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
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
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
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f'Warning: could not probe duration of {file_name}: {e}')
        return None


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
