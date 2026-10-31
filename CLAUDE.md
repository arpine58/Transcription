# Project rules (read at the start of every chat)

Goal: live Eastern Armenian → English transcription in the terminal, speaker labels `Me` / `Speaker N`,
≤ 5 s from end of phrase to printed line, all models local. Full context: PLAN.md, docs/TECH_PLAN.md, DECISIONS.md.

## Hard requirements (from the brief and Mathrix's answers)
- Python. Runs on the ThinkPad T16 (Core Ultra 7 255U, 32 GB, no NVIDIA GPU, Ubuntu 24.04, PipeWire).
- Local inference only; downloading open models at setup is fine.
- Transcription and translation are separate steps.
- Eastern Armenian only. Zoom in the interview. No assumption about the number of call-side speakers.
- Never commit model weights, audio or keys.

## How we work
- Keep it simple. No silent fallbacks, no catch-all `except`, no retries or timeouts "just in case".
  Fail loudly with a message that says what went wrong and what to do.
- One module = one job with a plain input and output (`audio`, `segmenter`, `asr`, `translate`, `speakers`,
  `pipeline`, `output`). Test a module on recorded audio before wiring it in.
- Model choices are made with numbers from `bench/` on the ThinkPad, never from impressions or model cards alone.
- Verify every model name, library API and parameter against its docs or by running it. Say where a claim comes from.
- Tests come from the requirements (TESTS.md), not from the code just written.
- Small commits, one change each. After three failed attempts at the same problem: stop, go back to the last
  working commit, shrink the problem to a small example.
- Every important decision goes into DECISIONS.md with its reason.
- Nothing is pushed without Arpine's review.

## Style
- Python 3.11+, type hints, dataclasses for plain data, short modules, comments only where the why is not obvious.
- CLI flags via argparse; no config files until they are needed.
