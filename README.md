# Audiblez: Generate  audiobooks from e-books

[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
[![Git clone and run](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

### v4 Now with Graphical interface, GPU acceleration (CUDA / AMD ROCm / Apple Silicon), and many languages!

![Audiblez GUI on MacOSX](./imgs/mac.png)

Audiblez generates `.m4b` audiobooks from regular `.epub` e-books,
using Kokoro's high-quality speech synthesis.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports these languages: 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳

On a Google Colab's T4 GPU via Cuda, **it takes about 5 minutes to convert "Animal's Farm" by Orwell** (which is about 160,000 characters) to audiobook, at a rate of about 600 characters per second.

On my M2 MacBook Pro, on CPU, it takes about 1 hour, at a rate of about 60 characters per second.


## How to install the Command Line tool

If you have Python 3 on your computer, you can install it with pip.
You also need `espeak-ng` and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install audiblez
```

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
pip install audiblez
pip install "audiblez[mlx]"                          # optional: native Apple Silicon (MLX) engine, fastest on Mac
```

Then you can convert an .epub directly with:

```
audiblez book.epub
```

This uses the default voice `af_heart` (the highest-quality English narrator). Pick a different
voice, a curated blend, or your own mix with `-v` (see [Supported Voices](#supported-voices)):

```
audiblez book.epub -v af_bella                 # a different high-quality voice
audiblez book.epub -v af_warm                  # a curated "house narrator" blend
audiblez book.epub -v 'af_bella:60,af_heart:40' # your own weighted blend
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## How to run the GUI

The GUI is a simple graphical interface to use audiblez.
You need some extra dependencies to run the GUI:

```
sudo apt install ffmpeg espeak-ng 
sudo apt install libgtk-3-dev        # just for Ubuntu/Debian 🐧, Windows/Mac don't need this
  
pip install "audiblez[ui]"
```

Then you can run the GUI with:
```
audiblez-ui
```

## How to run on Windows

After many trials, on Windows we recommend to install audiblez in a Python venv:

1. Open a Windows terminal
2. Create anew folder: `mkdir audiblez`
3. Enter the folder: `cd audiblez`
4. Create a venv: `python -m venv venv`
5. Activate the venv: `.\venv\Scripts\Activate.ps1`
6. Install the dependencies: `pip install "audiblez[ui]"`
7. Now you can run `audiblez` or `audiblez-ui`
8. For Cuda support, you need to install Pytorch accordingly: https://pytorch.org/get-started/locally/


## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
audiblez book.epub -v af_heart -s 1.5
```

## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

The default voice is **`af_heart`** — the highest-graded English narrator (Kokoro grades it **A**).
Voices vary a lot in quality (see the grades in `audiblez --help`); the best English ones are
`af_heart` (A), `af_bella` (A-), `bf_emma` (B-, British) and `af_nicole` (B-).

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more details about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

### Voice blends ("house narrators")

You can blend voices into a single, consistent narrator. Kokoro averages the voices, so a blend is
still a fixed preset (no voice cloning, no reference clip) and is just as stable over a whole book —
it works on every backend, including the Apple-Silicon MLX engine.

**Curated presets** (use the name directly with `-v`):

| Preset | Blend | Character |
|---|---|---|
| `af_warm` | Heart + Bella | warm, smooth natural female |
| `af_expressive` | Heart + Nicole | expressive, intimate female |
| `ab_storyteller` | Heart + Emma | neutral US/UK female storyteller |
| `am_warm` | Michael + Fenrir | warm male |
| `am_deep` | Puck + Onyx (2:1) | rich, low male baritone (~100 Hz) |

**Custom blends** — mix any voices with optional weights:

```
audiblez book.epub -v 'af_bella,af_heart'        # 50/50 blend
audiblez book.epub -v 'af_bella:60,af_heart:40'  # weighted 60/40 blend
audiblez book.epub -v 'af_heart:3,am_michael:1'  # mostly Heart with a touch of Michael
```

In the GUI, the recommended voices and curated blends appear at the top of the Voice dropdown
(each annotated with its quality grade), and you can type a custom blend like `af_bella:60,af_heart:40`
directly into the box. For an in-depth comparison of voices, blends, and alternative TTS models, see
[`docs/tts-naturalness-deep-dive.md`](docs/tts-naturalness-deep-dive.md).

## Choosing a backend (GPU acceleration)

By default audiblez runs on **CPU**. Use `-b/--backend` to pick a faster engine:

| Backend | Flag      | Hardware        | Notes                                                                       |
|---------|-----------|-----------------|-----------------------------------------------------------------------------|
| CPU     | `-b cpu`  | any             | default                                                                     |
| CUDA    | `-b cuda` | NVIDIA GPU      | requires a CUDA build of PyTorch                                            |
| ROCm    | `-b rocm` | AMD GPU         | Linux only; requires a ROCm build of PyTorch                                |
| MPS     | `-b mps`  | Apple Silicon   | PyTorch Metal backend                                                       |
| MLX     | `-b mlx`  | Apple Silicon   | native Apple engine, fastest on Mac — needs `pip install "audiblez[mlx]"`   |

```
audiblez book.epub -v af_heart -b mlx     # Apple Silicon (fastest)
audiblez book.epub -v af_heart -b cuda    # NVIDIA
audiblez book.epub -v af_heart -b rocm    # AMD (Linux)
```

The GUI exposes the same choices as radio buttons — only the backends actually available on your
machine are shown. If you request a backend that isn't available, audiblez warns and falls back to CPU.
`--cuda` is still accepted as a deprecated alias for `-b cuda` (and maps to `-b rocm` on a ROCm build).

**Apple Silicon** is fully supported via both **MLX** (native, fastest — `pip install "audiblez[mlx]"`)
and **MPS** (PyTorch Metal). On an M-series Mac, MLX narrates several times faster than CPU.

**NVIDIA Cuda example:** [Audiblez on a Google Colab Notebook with Cuda](https://colab.research.google.com/drive/164PQLowogprWQpRjKk33e-8IORAvqXKI?usp=sharing).

## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Help page

For all the options available, you can check the help page `audiblez --help`:

```
usage: audiblez [-h] [-v VOICE] [-p] [-s SPEED] [-b {cpu,cuda,rocm,mps,mlx}] [-o FOLDER] epub_file_path

positional arguments:
  epub_file_path        Path to the epub file

options:
  -h, --help            show this help message and exit
  -v, --voice VOICE     Narrating voice (default: af_heart). A voice id (e.g. af_heart),
                        a preset blend (af_warm, af_expressive, ab_storyteller, am_warm,
                        am_deep), or a custom blend like 'af_bella:60,af_heart:40'.
  -p, --pick            Interactively select which chapters to read in the audiobook
  -s, --speed SPEED     Set speed from 0.5 to 2.0
  -b, --backend {cpu,cuda,rocm,mps,mlx}
                        Narration backend: cpu, cuda (NVIDIA), rocm (AMD), mps
                        (Apple Silicon), mlx (Apple Silicon native). Default: cpu.
  -o, --output FOLDER   Output folder for the audiobook and temporary files

examples:
  audiblez book.epub -v af_heart           # best-quality default voice
  audiblez book.epub -v af_warm -b mlx     # a curated "house narrator" blend
  audiblez book.epub -v 'af_bella:60,af_heart:40'  # your own weighted blend

to use the GUI, run:
  audiblez-ui
```

## Author

by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)
