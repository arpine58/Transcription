# Live Armenian → English transcription

A Python terminal program that listens to a video call, transcribes spoken **Eastern Armenian**, translates each
phrase to **English** and labels who spoke: `Me` for the microphone, `Speaker 1`, `Speaker 2`, … for the people on
the call. Everything runs locally on the laptop: no cloud speech or translation APIs.

```
[00:00:21] Speaker 2  HY: Վաղը առավոտյան ժամը տասին հանդիպում ունենք։
                      EN: We have a meeting tomorrow morning at ten.
[00:00:27] Me         HY: Լավ եմ, շնորհակալություն։
                      EN: I'm fine, thank you.
```
(Illustrative layout; real output contains recognition errors, see section 6.)

**Honest summary.** Capture of both sides, speaker labels and the latency budget work on the target laptop
when one person speaks at a time. Armenian recognition is the weak link: on our own test clips the chosen model
gets about 1 in 5 letters wrong, so the English is often only roughly right. The numbers and the reasons are
below and in [DECISIONS.md](DECISIONS.md).

Target machine: Lenovo ThinkPad T16 Gen 4, Intel Core Ultra 7 255U, 32 GB RAM, integrated Intel GPU
(no NVIDIA GPU), Ubuntu 24.04 with PipeWire, Wayland.

---

## 1. Setup

### 1.1 System packages and Intel GPU driver

```bash
sudo apt install pulseaudio-utils            # pactl and parec (work on top of PipeWire)

# Intel GPU compute driver, so OpenVINO can use the integrated GPU (Intel's official PPA)
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
sudo gpasswd -a ${USER} render               # then log out and back in
clinfo | grep "Device Name"                  # should list Intel(R) Graphics
```

### 1.2 Python environment

Python 3.12. Install the **CPU** builds of torch first; otherwise pip pulls CUDA builds that fail to load
(`libcudart.so` missing).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-openvino.txt
python tools/check_openvino.py               # must list GPU
```

### 1.3 Models (one-time, about 5 GB, stored in `models/` and `~/.cache/huggingface`, never committed)

```bash
# speech recognition: Whisper large-v3 with int4 weights for OpenVINO
optimum-cli export openvino --model openai/whisper-large-v3 --weight-format int4 models/whisper-large-v3-int4-ov

# translation: NLLB-200 distilled 600M, converted to CTranslate2 int8
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
    --output_dir models/nllb-200-distilled-600M-ct2 --quantization int8
```

The voice-activity model (Silero) and the speaker model (SpeechBrain ECAPA) download automatically on first run.

### 1.4 Audio devices

No virtual loopback driver is needed. PipeWire exposes every output device as a **monitor** source, so the
program records exactly what you hear, and you still hear the call normally.

1. Plug in **headphones**. With laptop speakers the microphone also hears the call, which creates fake `Me` lines
   and doubles the work (echo handling is out of scope per the brief).
2. In Zoom, *Settings → Audio*: pick the headphones as speaker and your microphone as microphone.
3. List the devices:

   ```bash
   python -m livetx --list-devices
   ```
   ```
   Default mic:            alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source
   Default system monitor: bluez_output.E8_D0_3C_A5_43_A2.1.monitor
     51  alsa_output....monitor  [system audio (monitor)]
     52  alsa_input....source    [input]
   ```
   By default the program uses the default input as the mic and the monitor of the default output as the call.
   If Zoom plays into another device, pick it explicitly with `--system <index or name>` (and `--mic` likewise).
4. Check that both sides are captured: speak, and play something through the headphones.

   ```bash
   python tools/record_both.py --seconds 10   # both level meters should move
   ```

## 2. Run

```bash
python -m livetx                             # default devices, default models
python -m livetx --mic 52 --system 51        # explicit devices
```

Models load and warm up first, then it prints `listening.` Stop with Ctrl+C.

| Flag | Default | Meaning |
|---|---|---|
| `--asr-model` | `ov:models/whisper-large-v3-int4-ov@GPU` | `ov:<dir>@GPU` (OpenVINO) or a faster-whisper name for CPU |
| `--min-silence` | `0.4` | seconds of silence that end a phrase |
| `--max-segment` | `6.0` | a longer phrase is cut and printed in parts |
| `--speaker-threshold` | `0.45` | voice similarity needed to count as the same call speaker |
| `--no-speakers` | off | label the call side just `Call` |
| `--save-segments DIR` | off | save every phrase as WAV, to hear what the recogniser got |
| `--latency-log` | `latency.tsv` | per-line timings, see below |

Lines the recogniser could not handle are shown, not hidden: `(unclear: not Armenian script)`,
`(unclear: repetitive output)`. If transcription falls behind, `[behind by N s, M segments waiting]` is printed.

`latency.tsv` has a header row. Columns: phrase start, speaker, phrase length, voiced seconds (call side),
phrases joined into one Whisper call, queue wait, ASR s, speaker ID s, translation s, **total s from end of phrase
to printed line**, speaker similarity (call side), result (`ok` or `unclear: reason`). Delete the old file before a
run, since the column layout changed on Oct 31.

Tests: `pytest` (24 unit tests, no models needed).

---

## 3. How it works

```
 mic source ───────► parec ─► Segmenter (Silero VAD) ─┐
                                                      ├─► one queue ─► Whisper (GPU) ─► checks ─► NLLB ─► speaker ─► terminal
 output .monitor ──► parec ─► Segmenter (Silero VAD) ─┘                 Armenian text            English   label
```

| Module | In → out |
|---|---|
| `livetx/audio.py` | PipeWire source name → 30 ms float32 blocks at 16 kHz (one `parec` process per source) |
| `livetx/segmenter.py` | blocks → phrases: closes after 0.4 s silence or at 6 s, 0.2 s pre-roll, drops < 0.5 s |
| `livetx/asr.py` | phrase audio → Armenian text + checks (Armenian script ≥ 80 %, compression ratio ≤ 2.4, output length cap) |
| `livetx/translate.py` | Armenian text → English text (NLLB `hye_Armn → eng_Latn`) |
| `livetx/speakers.py` | phrase → `Me` (mic) or `Speaker N` (192-dim ECAPA embedding + online clustering) |
| `livetx/pipeline.py` | threads, queue, latency timestamps |
| `livetx/output.py` | terminal line formatting |

Design choices, each with its reason in DECISIONS.md:

- **Two capture processes, one recogniser.** Both sources are captured in parallel, but there is a single GPU, so
  phrases are transcribed one at a time in arrival order. Two recognisers on one GPU would only slow each other.
  Speaker labels and translation run on the CPU while the GPU works on the next phrase. When phrases pile up,
  the waiting phrases of each side are joined into one Whisper call, because each call has a fixed cost of about 2 s.
- **Transcription and translation are separate steps** (Mathrix's requirement), so the Armenian text is shown
  and can be measured on its own.
- **Speakers.** Each call-side phrase gets a voice fingerprint (of its voiced part only) and joins the most
  similar known speaker if the cosine similarity is above 0.45, otherwise it starts a new speaker. Labels are
  never renamed, so a person keeps their label for the session. Phrases with under 1 s of speech are too short
  to fingerprint and get the previous call speaker; only 1.5 s of speech or more can start a new speaker; two
  speakers whose voices turn out to be the same are merged. `tools/speaker_replay.py` replays saved phrases
  through the tracker offline and scores the labels.
- **Fail loudly.** A dead audio stream stops the program with an error instead of going silent; unreliable
  recognition is printed as unclear instead of as a confident wrong translation.

## 4. Model choices

All numbers were measured on the ThinkPad with `bench/asr.py` on our own 12 clips of Eastern Armenian
(`eval/refs.tsv`, sentences in `eval/sentences.md`). Letter error rate (CER) is the fairer number for Armenian:
one wrong letter makes a whole word count as wrong, so word error rates (WER) look very high for every model.

### Speech recognition

| Model | Where | Letters wrong (CER) | Time vs speech | Verdict |
|---|---|---|---|---|
| Whisper large-v3-turbo | CPU | 95 % (1 clip) | ~3.5× | writes Armenian in Latin letters; dropped |
| Whisper large-v3 int8 | CPU | 15 % (1 clip) | ~5.6× (15 s per clip) | far too slow; thread count did not help |
| Whisper large-v3 int8 | **GPU** | **16 %** | 1.03× | most accurate, but slower than speech |
| **Whisper large-v3 int4** | **GPU** | **20 %** | **0.78×** | **chosen**: nearly as accurate, faster than speech |
| Whisper medium int8 | GPU | 27 % | 0.54× | fast, but often phonetic mush |
| Edmon02/whisper-small-hy | CPU | 100 % (empty output, 1 clip) | — | dropped |
| BuzzASR/armenian (Armenian fine-tune of large-v3) | CPU / GPU | empty output | — | dropped after 3 attempts, see below |

CER for the GPU rows is on the 10 pure-Armenian clips, excluding one clip on which both large models get stuck
repeating a syllable (that failure is caught live and shown as unclear). "Time vs speech" below 1 means it
transcribes faster than people talk.

Why the GPU: the CPU cannot keep up with any large model, and the integrated GPU (OpenVINO GenAI) made large-v3
4.5× faster. Why int4: on an integrated GPU, writing out the text is limited mostly by how fast the weights are
read from memory, so smaller weights decode faster; it cost 4 points of CER and gained the margin needed to stay
ahead of speech.

Considered and not used:

- **BuzzASR/armenian** reports 11.9 % WER on FLEURS on its model card, the most promising option. It replaces
  Whisper's tokenizer (no language tokens), so OpenVINO refuses it, and with transformers on CPU it returned empty
  text for every clip. Unverified on our hardware; first thing to retry with more time (the transformers
  benchmark backend used for it was removed in the cleanup and would need to be added back).
- **SeamlessM4T v2** supports Armenian speech, but is heavy on CPU and its main strength (direct speech-to-English)
  is ruled out by the separate-steps requirement.
- **An Armenian hint for Whisper** (initial prompt): with one sentence it removed the looping failure, with
  another it made long phrases stop early (CER 39 %). Too unstable to keep; removed from the code.

### Translation

`facebook/nllb-200-distilled-600M`, CTranslate2 int8 on CPU: 0.2 to 0.9 s per phrase, never the bottleneck.
NLLB-200 covers Armenian (`hye_Armn`) and runs offline. Not scored separately: its quality is limited mostly by
the errors in the Armenian it receives. The 1.3B NLLB is the obvious next step if recognition improves.

### Speakers

`speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, freely downloadable, no account). Chosen over pyannote, which is
gated behind a Hugging Face token and heavier. In the Zoom test a new voice scored 0.29 against the existing
speaker and correctly became Speaker 2; later phrases matched their own speaker at 0.65 to 0.88.

## 5. Measured results

| Check | Result on the ThinkPad |
|---|---|
| Mic and call captured at the same time | yes |
| One side speaking: end of phrase → printed line | 2.5–5.8 s (Zoom test with medium); large-v3 int4 still to be measured live |
| Both sides speaking at once | queue builds up, lines 9–14 s late |
| Call-side speakers kept apart | yes in the Zoom test (similarity 0.29 vs 0.65–0.88) |
| Armenian recognition (CER, eval clips) | 20 % |

## 6. Known limitations

- **Armenian accuracy.** About 1 in 5 letters wrong. Short phrases are worst (`Բարև` heard as `Պարև`), which also
  makes the English for greetings unreliable.
- **Latency when both sides talk.** One GPU, one queue: overlapping speech falls behind and catches up in pauses.
- **Echo.** Without headphones the mic hears the call. Out of scope per the brief.
- **Speaker threshold** is tuned on one short Zoom test; very similar voices may merge, a noisy line may split one
  person into two.
- **Mixed Russian/English words** are not handled (`meeting` came out as `Միտինգ`).
- **Small evaluation set**: 12 clips from one speaker. Enough to rank models, not to quote precise error rates.
- **Overlapping speakers** inside one phrase get one label.

## 7. With more time

1. Get an Armenian fine-tuned Whisper running on the GPU (BuzzASR, or fine-tune large-v3 on Common Voice Armenian
   and FLEURS), the biggest lever on quality.
2. A second recogniser on the NPU for the mic side, so both sides are transcribed in parallel.
3. A larger, multi-speaker evaluation set, including real Zoom audio, and a translation score (e.g. chrF against
   reference English).
4. Streaming partial results for long phrases instead of fixed 6 s cuts.

---

## Repository

```
livetx/      the program (see section 3)
bench/       model benchmark: bench/asr.py, Armenian text normalisation
eval/        reference texts for the test clips (the audio itself is not committed)
tools/       record_both.py, record_clip.py, check_openvino.py, speaker_check.py, speaker_replay.py
tests/       unit tests, written from the brief's requirements
AI_WORKFLOW.md  how the AI assistant was used, what it got wrong, how it was checked
docs/        the original technical plan
PLAN.md      the plan we started from · DECISIONS.md  every decision with its reason
TESTS.md     checks derived from the brief · CLAUDE.md  project rules for the AI assistant
```

Reproduce the model comparison:

```bash
pip install -r bench/requirements.txt
python tools/record_clip.py clip01 --speaker yourname     # record clips and their exact text
python bench/asr.py ov:models/whisper-large-v3-int4-ov@GPU ov:models/whisper-medium-ov@GPU
```
