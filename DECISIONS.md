# DECISIONS.md

The decisions that shaped the project, each with its reason. Numbers are measured on the ThinkPad
(bench/asr.py on 12 recorded clips, latency.tsv from live calls). Format: decision · why.

## Scope
- Eastern Armenian only, Zoom, 20–30 h budget; no limit assumed on the number of call-side speakers. · Answers to our clarification questions.
- Transcription and translation are separate steps. · Required; also lets us show and measure the Armenian text on its own.
- Mixed Russian/English words are a stretch goal. · Core Armenian quality first.

## Audio
- Call audio comes from the PipeWire `.monitor` of the output device. · Built into Ubuntu 24.04, no extra driver, and you still hear the call.
- One `parec` process per source (16 kHz mono), not sounddevice. · sounddevice cannot select a PipeWire monitor by name; `parec` can, and two processes capture both sides at once.
- A source that stops unexpectedly raises an error. · A silently dead stream would look like "nobody is talking".
- Phrases are cut with Silero VAD and a small state machine: 0.4 s of silence ends a phrase, 6 s hard cap, 0.2 s pre-roll, under 0.5 s of speech is dropped. · Easy to read and tune; the cap keeps long monologues inside the 5 s budget.

## Speech recognition
- Whisper large-v3 with int4 weights on the Intel GPU through OpenVINO GenAI. · The only option that is both accurate and faster than speech:

  | Model (GPU, 10 Armenian clips) | Letters wrong (CER) | Time vs speech |
  |---|---|---|
  | large-v3 int8 | 16 % | 1.03x |
  | **large-v3 int4** | **20 %** | **0.78x** |
  | medium int8 | 27 % | 0.54x |

  Arpine chose "large-v3, tuned for speed" over medium. On an integrated GPU, decoding is limited by reading the
  weights from memory, so 4-bit weights decode faster.
- The GPU, not the CPU. · large-v3 took ~15 s per clip on the CPU whatever the thread count (4, 8, 14); the GPU was 4.5x faster. Needs Intel's GPU driver (PPA kobuk-team/intel-graphics).
- Whisper settings: language forced to Armenian, beam 1, no conditioning on earlier text, output capped at 10 + 25 tokens per second of audio. · Phrases are independent; conditioning causes repeated hallucinations; the cap stops a looping decode in about a second.
- Unreliable output is shown as `(unclear: reason)` and not translated: under 80 % Armenian letters, compression ratio above 2.4, or low confidence (CPU backend only). · Whisper hallucinates on weak languages; an honest "unclear" beats a confident wrong translation.
- Errors are scored as CER, with Armenian normalisation (՞ ՜ ՛ deleted, other punctuation becomes a space). · One wrong letter fails a whole Armenian word, so WER is too harsh; without normalisation "ինչպե՞ս" counts as two words.

## Translation and speakers
- NLLB-200 distilled 600M, CTranslate2 int8 on the CPU. · Covers Armenian, runs offline, 0.2–0.9 s per phrase, never the bottleneck.
- Mic is always `Me`. Call side: SpeechBrain ECAPA voice fingerprints, online clustering with cosine threshold 0.45; phrases under 1 s keep the last call speaker; labels never change. · Simple and explainable, no Hugging Face token (pyannote is gated and heavier). In the Zoom test a new voice scored 0.29 and known voices 0.65–0.88, so 0.45 separates them.
- Speaker tracker repaired instead of replaced (Oct 31): fingerprint only the voiced part of a phrase (Silero VAD), at least 1 s of speech to fingerprint at all, 1.5 s to start a new speaker, only confident matches update a voice, speakers that turn out alike are merged; no cap on the number of speakers. `tools/speaker_replay.py` scores settings offline on saved phrases. · In the Oct 30 call one person became 8 speakers: 6 of the 7 extra speakers came from phrases under 2 s, often under 1 s of real speech, while longer phrases matched one person. pyannote and NeMo Sortformer were considered and not used: unmeasured on a CPU-only laptop, need a Hugging Face token or NVIDIA-oriented tooling, and risk the working setup. Threshold and the 1.5 s value still to be set from a recorded test call.

## Pipeline
- One ASR worker on the GPU handles both sides in arrival order, with a visible "behind by N s" warning; speaker labels and translation run in a second worker on the CPU at the same time. · One GPU, so one recogniser; the CPU steps no longer wait their turn.
- When phrases pile up, the waiting phrases of each source are joined (up to 15 s) into one Whisper call. · Second Zoom call: each call cost 2.2 s plus 0.5 s per second of audio (Whisper always encodes 30 s), so 33 short phrases needed 141 s of work for 130 s of call and lines fell 30 s behind. Replaying that log with joining and the parallel CPU step: median 6.9 s, worst 11.9 s instead of 18.6 s and 33 s; to be confirmed on a call.
- NLLB output capped at 1.5x the input tokens + 10. · On Oct 31 one translation took 9.1 s, most likely running on to the 256-token limit.
- Every line is logged to latency.tsv, including unclear lines, with speaker ID and translation timed separately and the number of joined phrases. · Delays are measured, not guessed; unclear lines also use the GPU, so leaving them out under-counted the load.

## Setup
- torch 2.8.0 and torchaudio 2.8.0, CPU builds, pinned. · pip otherwise pulls NVIDIA builds that fail to load (`libcudart` missing); torchaudio 2.9 removed what SpeechBrain imports.
- Every model step is tested on the ThinkPad before it counts as working. · Only the real laptop shows real speed and real errors.

## Tried and dropped
- **large-v3-turbo.** On the CPU, short phrases came out as English, Tibetan or Cyrillic. Re-tested on Oct 31 on the GPU (int4) for speed: twice as fast as large-v3 int4 (real-time factor 0.48 vs 0.93, all 12 clips), but 5 of 12 clips came out in Latin letters and one looped (CER 48 % vs 44 %). Short phrases would show as unclear, so not used.
- **Edmon02/whisper-small-hy.** Empty output.
- **BuzzASR/armenian** (Armenian fine-tune, 11.9 % WER on its model card). OpenVINO refused its replaced tokenizer, and transformers on CPU returned empty text for every clip. Stopped after three attempts; the first thing to retry with more time.
- **medium as the live model.** It kept up on a Zoom call (2.5–5.8 s per line), but the Armenian was poor. Replaced by large-v3 int4.
- **An Armenian hint sentence for Whisper.** The first sentence matched two test clips word for word, so its gain was unfair. A neutral sentence made results worse (CER 39 %). A gain that flips with the sentence is noise, so the code was removed.
- **Code for the dropped items** (hint sentence, transformers benchmark backend, BuzzASR special case) was removed in the Oct 30 cleanup. · Less code to explain and test; the results stay in the README and bench/results/.
