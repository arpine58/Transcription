# Technical plan: live Armenian → English transcription

> Original technical plan from Oct 28, kept for the record. The built system differs: see README.md (what it is) and DECISIONS.md (why each change).

Draft v2, Oct 28 (updated with Mathrix's answers, see section 0). Companion to PLAN.md. Target machine: ThinkPad T16 Gen 4, Core Ultra 7 255U, 32 GB, no NVIDIA GPU,
Ubuntu 24.04 (PipeWire). Everything below runs on CPU unless step 1 proves otherwise.

## 0. Clarified requirements (answers from Mathrix)

- Budget 20–30 hours. Submitting before Nov 1 is fine.
- Interview call runs on **Zoom**. On Linux, Zoom plays into the selected output, so we capture that output's `.monitor`.
- **Eastern Armenian only.**
- Any number of people on the call side; clustering stays general (Speaker N).
- Russian/English words mixed in: expected, but can wait until the core works (stretch goal, step 4).
- 5 s is measured from the end of each phrase to its line. Long monologues may be split into several lines.
- Downloading open-source models at setup is fine; inference must be local.
- Transcription and translation must be **separate steps** (no direct speech-to-English model).
- `chats/`: one Markdown file per Cowork thread.

## 1. Pipeline

```
 mic (PipeWire source) ─┐                                    ┌─ source=mic    → "Me"
                        ├─► capture ─► VAD segmenter ─► queue ─► ASR (hy text) ─► MT (en text) ─► printer
 sink monitor ──────────┘   (2 threads,  (per source)      │                                    ▲
 (call audio)               16 kHz mono)                   └─ source=system → speaker embedding ─┘
                                                                              + online clustering → "Speaker N"
```

- One capture thread per source (`sounddevice` InputStream, 16 kHz mono float32, 30 ms blocks).
- Segmenter per source: Silero VAD. A segment closes after ~500 ms of silence or at a hard cap (~8 s) so a
  monologue cannot blow the latency budget.
- One worker thread runs ASR then MT on segments in order (models are CPU-heavy; two parallel ASR runs would
  just fight for cores). The speaker step runs on the same segment before printing.
- Each segment carries timestamps (`speech_end`, `asr_done`, `mt_done`, `printed`) so latency is logged, not guessed.

## 2. Audio capture on Ubuntu

- PipeWire exposes every output as a "monitor" source. The call audio is `<your speakers/headset>.monitor`,
  and you still hear the call normally. No BlackHole / loopback driver needed.
- `python -m livetx --list-devices` prints devices; `--mic <id|name>` and `--system <id|name>` pick them.
- Check: `pactl list short sources` shows the `.monitor` entry. If `sounddevice` can't see it, route through the
  `pipewire`/`pulse` ALSA device and pick the source with `PULSE_SOURCE`, or capture with `parec`. We confirm which
  works on your laptop in step 1, before anything else is built.
- Headphones in the demo avoid echo (the brief says echo handling is out of scope, but it will still pollute "Me").

## 3. Models (candidates, chosen by measurement in step 1)

**ASR (Armenian speech → Armenian text).** The heart of quality and latency.
| Candidate | Size | Notes |
|---|---|---|
| `openai/whisper-large-v3` / `large-v3-turbo` via faster-whisper int8 | 1.5B / 0.8B | Baseline. Armenian is supported but weaker than major languages. |
| `BuzzASR/armenian` | large-v3 fine-tune | Model card reports 11.9% WER on FLEURS hy, MIT license. Big, so CPU speed is the question; custom tokenizer may complicate CTranslate2 conversion. |
| `Edmon02/whisper-small-hy` | small (244M) | Fast fallback, quality to be measured. |
| `facebook/seamless-m4t-v2-large` | 2.3B | Supports Armenian speech input and text output (used as ASR only, since steps must be separate). Heavy on CPU. |

Rule: pick the most accurate model whose p90 processing time per segment leaves room in the 5 s budget.

**Translation (hy → en).**
- `facebook/nllb-200-distilled-600M` (`hye_Armn → eng_Latn`), converted to CTranslate2 int8: expected well under 1 s per sentence on CPU.
- Alternatives to compare: Seamless text-to-text, NLLB 1.3B if 600M quality is poor.
  Direct speech-to-English (Whisper `task="translate"`) is ruled out: Mathrix wants separate steps.

**Speakers (call side only).**
- `speechbrain/spkrec-ecapa-voxceleb` embeddings (192-dim) per segment.
- Online clustering: compare to each known speaker's running mean embedding; cosine similarity above a threshold → same
  speaker, else new `Speaker N`. Segments under ~1 s get the previous speaker's label rather than a fresh guess.
- Threshold tuned on our own two-voice recordings. Alternative: pyannote (heavier, needs a Hugging Face token).
- Mic side is always `Me`, no model needed.

All model names above were checked on Hugging Face; nothing gets used before it actually runs on your laptop.

## 4. Latency budget (end of phrase → line on screen, target ≤ 5 s)

| Stage | Budget |
|---|---|
| VAD silence to close the segment | 0.5 s |
| Queue wait | ≤ 0.5 s |
| ASR | ≤ 2.5 s |
| Speaker embedding | ≤ 0.2 s |
| Translation | ≤ 0.5 s |
| Slack | ~0.8 s |

If ASR misses its budget: smaller/turbo model, shorter max segment, fewer beams (beam_size 1–2), more CPU threads,
OpenVINO on the Intel GPU/NPU as a later experiment.

## 5. Repository layout

```
livetx/
  __main__.py      # CLI: --list-devices, --mic, --system, --asr-model, ...
  audio.py         # device listing, capture threads → frames
  segmenter.py     # Silero VAD → Segment(source, start, end, audio)
  asr.py           # Transcriber.transcribe(audio) -> str
  translate.py     # Translator.translate(hy) -> str
  speakers.py      # SpeakerTracker.label(segment) -> "Me" | "Speaker N"
  pipeline.py      # threads + queue, latency timestamps
  output.py        # terminal formatting
bench/             # scripts that compare models on eval clips (WER, seconds)
eval/              # clip list + reference texts only (audio itself is NOT committed)
tests/             # pytest, written from the brief's requirements
chats/             # exported Claude chats, one .md per session
README.md PLAN.md DECISIONS.md TESTS.md requirements.txt .gitignore
```

Each module has one small class with a plain input and output, so it can be tested on recorded audio on its own.
Errors fail loudly (no silent fallbacks), per the tips.

## 6. Evaluation set

- ~10–15 short Eastern Armenian clips (3–10 s), each with the correct text in `eval/refs.tsv`.
- Mix: you speaking into the mic, 2+ other voices (friends/family, or call recordings with consent), a little background noise,
  a couple of clips with Russian/English words mixed in.
- One 2–3 minute recorded two-person conversation, played through the speakers, for speaker-label and end-to-end testing.
- Metrics: WER/CER (`jiwer`) for ASR, a quick 1–5 rating of English for MT, speaker label accuracy, p50/p90 latency.

## 7. Steps in detail

1. **Foundation and measurement (~5 h)**
   - Create repo, `.gitignore` (weights, audio, `.env`), PLAN/DECISIONS/TESTS files.
   - Capture script: record 10 s from mic and monitor at once into two WAVs. This proves the audio setup.
   - Record the eval clips and write reference texts.
   - `bench/asr.py`: run each ASR candidate over the clips → WER + seconds per clip table. Same for MT.
   - Write the decision with numbers into DECISIONS.md.
2. **Walking skeleton (~5 h)**: audio → segmenter → ASR → MT → terminal with `Me` / `Call` labels; latency logged.
3. **Speakers + latency (~5 h)**: SpeakerTracker, tuning on the two-voice recording, then a real call with two people.
4. **Harden + docs + rehearsal (~4 h)**: pytest from requirements, README, chat export, full dress rehearsal on Zoom with call-side people.
   Stretch if time allows: better handling of Russian/English words inside Armenian speech.

## 8. What I need from you

1. **A GitHub repo.** Create an empty repo (e.g. `armenian-live-transcription`) on github.com and tell me its name.
   I can't create repos myself, but I can then push code to it. If it asks, [Connect GitHub](https://claude.ai/connect-github).
2. **Running things on your laptop.** I work in a cloud machine with no microphone and no speakers, so anything with
   real audio or real speed must run on your ThinkPad. Two options:
   - you run the commands I give you and paste the output back, or
   - you approve a Remote Control session, where a Claude session runs commands on your laptop directly (faster).
3. **Basic setup on the laptop:** Python 3.11 or 3.12, `git`, about 15 GB free disk for models, a headset.
4. **Armenian recordings + their correct text** (section 6). I can't produce real Armenian speech, so this is yours:
   ~30 minutes of recording with 2–3 people.
5. **Two people for a test Zoom call** before the interview, ideally from separate computers.
6. **Your review.** Read what I write and push back; the evaluation is about you steering, so your edits to
   this plan and DECISIONS.md are part of the deliverable.
