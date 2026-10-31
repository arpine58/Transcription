"""Compare speech recognition models on the eval clips: WER, CER and speed on this machine.

    python bench/asr.py ov:models/whisper-large-v3-int4-ov@GPU fw:large-v3

Model specs (the same transcribers the live program uses, see livetx/asr.py):
  fw:<name or path>   faster-whisper (CTranslate2, int8 on CPU), e.g. fw:large-v3, fw:large-v3-turbo
  ov:<dir>[@DEVICE]   OpenVINO GenAI, exported with optimum-cli; DEVICE is GPU (default), NPU or CPU

Writes per-clip transcripts to bench/results/<model>.tsv and prints a summary table.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.textnorm import normalize  # noqa: E402
from livetx.audio import read_wav  # noqa: E402

REFS = ROOT / "eval" / "refs.tsv"
CLIPS = ROOT / "eval" / "clips"
RESULTS = ROOT / "bench" / "results"


def load_refs() -> list[tuple[str, str]]:
    rows = []
    for line in REFS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        clip, _speaker, text = line.split("\t")
        rows.append((clip, text))
    if not rows:
        sys.exit(f"No clips in {REFS}. Record some with tools/record_clip.py first.")
    return rows


def make_transcriber(spec: str, threads: int):
    from livetx import asr

    kind, _, name = spec.partition(":")
    if kind == "fw":
        tr = asr.Transcriber(name, threads=threads)
    elif kind == "ov":
        tr = asr.make_transcriber(spec)
    else:
        raise ValueError(f"Unknown model spec '{spec}'. Use fw:<name> or ov:<dir>[@DEVICE].")
    return lambda x: tr.transcribe(x).text


def bench(spec: str, refs: list[tuple[str, str]], threads: int) -> dict:
    t0 = time.perf_counter()
    run = make_transcriber(spec, threads)
    load_s = time.perf_counter() - t0
    # a bare name lives in eval/clips/; a path with "/" is relative to the repo root (e.g. recordings/x.wav)
    clips = [(c, t, read_wav(ROOT / c if "/" in c else CLIPS / c)) for c, t in refs]
    run(clips[0][2])  # warm-up, not timed

    hyps, refs_n, secs, durs = [], [], [], []
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (re.sub(r"[^\w.-]", "_", spec) + ".tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("clip\tseconds\treference\thypothesis\n")
        for clip, ref, x in clips:
            t = time.perf_counter()
            hyp = run(x)
            dt = time.perf_counter() - t
            f.write(f"{clip}\t{dt:.2f}\t{ref}\t{hyp}\n")
            hyps.append(normalize(hyp))
            refs_n.append(normalize(ref))
            secs.append(dt)
            durs.append(x.size / 16_000)
    return {
        "model": spec,
        "wer": jiwer.wer(refs_n, hyps),
        "cer": jiwer.cer(refs_n, hyps),
        "p50": float(np.percentile(secs, 50)),
        "p90": float(np.percentile(secs, 90)),
        "rtf": sum(secs) / sum(durs),
        "load": load_s,
        "file": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    refs = load_refs()
    print(f"{len(refs)} clips, {args.threads} CPU threads\n")
    print("| model | WER | CER | p50 s/clip | p90 s/clip | real-time factor | load s |")
    print("|---|---|---|---|---|---|---|")
    for m in args.models:
        r = bench(m, refs, args.threads)  # printed per model, so a later crash doesn't lose earlier results
        print(f"| {r['model']} | {r['wer']:.1%} | {r['cer']:.1%} | {r['p50']:.2f} | {r['p90']:.2f} | {r['rtf']:.2f} | {r['load']:.0f} |", flush=True)
    print("\nPer-clip transcripts in", RESULTS.relative_to(ROOT))


if __name__ == "__main__":
    main()
