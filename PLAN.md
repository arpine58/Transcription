# PLAN.md — Live Armenian → English transcription with speaker labels

Draft v1 (Oct 28). Deadline Nov 1, aim to submit Oct 31.

> **Status Oct 29:** steps 1–3 done; step 4 (docs, chat export, rehearsal) in progress. This is the plan as written at the start, kept unchanged on purpose. What was actually built, and why it differs (Intel GPU via OpenVINO instead of CPU, large-v3 int4 instead of the candidates below), is in README.md and DECISIONS.md.

## Goal
A Python terminal program that captures the microphone and the system audio of a call at the same time,
transcribes Armenian, translates each segment to English, labels speakers (`Me`, `Speaker 1`, `Speaker 2`…)
and prints each line within 5 seconds of the phrase ending. All models run locally.

## Target machine
ThinkPad T16 Gen 4, Intel Core Ultra 7 255U (12 cores / 14 threads), 32 GB RAM, Intel Arc iGPU + NPU,
Ubuntu 24.04, Wayland (PipeWire audio). No NVIDIA GPU, so everything must run fast on CPU.

Verdict: sufficient, with care. 32 GB fits all models comfortably. The risk is ASR speed on CPU:
we use int8-quantized models (faster-whisper / CTranslate2) and measure latency on this laptop in step 1.
If CPU is too slow: smaller/turbo model, OpenVINO on the iGPU, or as a last resort a laptop with an NVIDIA GPU.
Bonus of Linux: PipeWire exposes a "monitor" of the speaker output, so no BlackHole-style setup is needed.

## Modules (input → output)
| Module | In | Out |
|---|---|---|
| `audio` | device ids (mic, sink monitor) | 16 kHz mono frames per source |
| `segmenter` (VAD) | frames | speech segments (start, end, audio, source) |
| `asr` | segment audio | Armenian text |
| `translate` | Armenian text | English text |
| `speakers` | segment audio + source | `Me` or `Speaker N` (stable per session) |
| `main` | CLI flags | terminal lines `[hh:mm:ss] Speaker  HY: …  EN: …` |

Candidates to evaluate (to be verified, not assumed):
- VAD: Silero VAD.
- ASR: Whisper large-v3 and large-v3-turbo (faster-whisper int8), community Armenian fine-tunes of Whisper on
  Hugging Face, Meta SeamlessM4T v2. Armenian is low-resource, so the choice is made by WER on our own clips.
- Translation: NLLB-200 distilled 600M (`hye_Armn → eng_Latn`) via CTranslate2; compare with Whisper/Seamless direct translation.
- Speakers: SpeechBrain ECAPA embeddings + simple online clustering (cosine threshold) on the call side only.

## Steps
1. **Foundation and measurement (~5 h).** Git repo, PLAN.md, DECISIONS.md, TESTS.md. Prove capture of mic and
   system audio together on Ubuntu. Record ~10 short Armenian clips with correct text (incl. 2+ voices).
   Benchmark ASR and translation candidates on the laptop: WER and seconds per segment. Pick models with numbers.
2. **Walking skeleton (~5 h, can start before the benchmark finishes; see DECISIONS.md).** Both sources → VAD → ASR → translation → terminal, with `Me` vs `Call` labels.
   Device listing and `--mic` / `--system` flags. Measure end-of-phrase → printed-line latency.
3. **Speakers and the 5 s budget (~5 h).** Split the call side into Speaker 1/2/… with embeddings and online
   clustering; test on a real call with two people. Tune VAD silence, chunk length and threading until latency holds.
4. **Harden, document, rehearse (~4 h).** Tests from the requirements, failure messages that are loud and clear,
   README (setup incl. audio, model choices and alternatives, architecture, limitations), export all chats to
   `chats/`, full dress rehearsal on a real video call.

## Risks
- Armenian ASR quality on CPU-sized models → measure early, keep a fallback model.
- Latency on CPU → int8, small segments, translation is cheap; measure every step.
- Speaker clustering with short segments → minimum segment length for embeddings, conservative threshold.
- Overlapping speech and Russian/English code-switching → document as known limitations.

## How we test
Eval clips with reference text (WER), timestamps logged per stage (latency), a scripted two-voice clip played
through the speakers' monitor (speaker labels), plus a live call rehearsal before the interview.
