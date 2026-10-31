# TESTS.md — checks derived from the brief

Status: [ ] open, [x] passing. Each check comes from a line of the brief, not from the code.

## Audio
- [x] Mic and system audio are captured at the same time (`tools/record_both.py` on the ThinkPad, both meters moved).
- [x] The user still hears the call while it is captured (monitor source, Zoom test).
- [x] `python -m livetx --list-devices` lists inputs and monitors; `--mic` / `--system` pick them; an unknown device fails with a clear message (`tests/test_audio.py`).

- [ ] (Not yet tried) An audio source that dies mid-session stops the program with a clear error (unplug the Bluetooth headset during a run).

## Evaluation
- [x] WER normalisation keeps Armenian words whole (`tests/test_textnorm.py`).
- [x] `bench/asr.py` gives WER, CER and seconds per clip for each model on the ThinkPad (results in DECISIONS.md).

## Transcription / translation
- [x] Armenian is transcribed: error rates on `eval/` clips recorded in DECISIONS.md per model (chosen model: 20% CER).
- [ ] Each segment gets an English translation that a reader understands. **Partly**: depends on recognition quality; not scored.
- [x] Transcription and translation are separate steps (Mathrix requirement): `asr.py` then `translate.py`.

- [x] Non-Armenian or repetitive ASR output is shown as "unclear", never translated (`tests/test_asr_checks.py`, cases from the first live run).

## Speakers
- [x] Everything from the mic is labelled `Me` (`tests/test_speakers.py`).
- [x] Clustering keeps two distinct voices apart with stable labels (unit test, synthetic embeddings).
- [x] Two real call-side voices on Zoom got `Speaker 1` and `Speaker 2` and kept them (one Zoom test, similarity 0.29 vs 0.65–0.88). Re-check in the rehearsal.

## Output and latency
- [x] Each line shows time, speaker, HY and EN (`tests/test_output.py`).
- [ ] ≤ 5 s from end of phrase to printed line. **Partly**: 2.5–5.8 s with one side talking (medium); 9–14 s when both talk. Re-measure with large-v3 int4.
- [x] Long monologues are split into several lines instead of waiting (6 s cap, seen in latency.tsv).

## Skeleton smoke checks
- [x] `python -m livetx` loads all models and prints "listening".
- [x] Speaking into the mic prints a `Me` line with HY and EN.
- [x] Call audio prints call-side lines (now `Speaker N`).
- [ ] `latency.tsv` total column stays under 5 s. **Partly**, see Output and latency.

## Constraints
- [ ] Runs fully offline after model download (test with Wi-Fi off).
- [ ] No weights, audio or keys in git (`.gitignore`, check `git status` before each commit).

## Before the interview (rehearsal checklist)
- [ ] Full Zoom call with remote people, headphones on, default settings; latency.tsv kept as evidence.
- [ ] Program started from a fresh terminal following only the README.
- [ ] Wi-Fi off after loading: still works.
- [x] AI_WORKFLOW.md explains how the AI assistant was used (replaces the per-chat export).
