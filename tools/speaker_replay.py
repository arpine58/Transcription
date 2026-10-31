"""Replay saved call phrases through the speaker tracker, offline, and score the labels.

    python -m livetx --save-segments segs_one          # during a call: saves every phrase as WAV
    python tools/speaker_replay.py segs_one             # labels the tracker would give
    python tools/speaker_replay.py segs_one --truth truth.tsv --threshold 0.3 0.4 0.45 0.5

truth.tsv (optional) says who really spoke each phrase, one line per file: `<file name>\t<person>`.
With it, each threshold gets two scores:
  labels per person   how many labels one real person got (1 is right)
  wrong lines         lines whose label is not the label that person mostly got (0 % is right)
Uses the same tracker, embedding model and voice trimming as the live program.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetx.audio import read_wav  # noqa: E402
from livetx.segmenter import Segment  # noqa: E402
from livetx.speakers import SpeakerTracker, ecapa_embedder, silero_trimmer  # noqa: E402


def score(labels: list[str], truth: list[str]) -> tuple[dict[str, int], float]:
    """Labels per real person, and the share of lines not labelled with their person's own label.

    Each person's own label = the label they got most often, each label given to at most one person
    (people with more lines choose first)."""
    per_person = {p: Counter(l for l, t in zip(labels, truth) if t == p) for p in dict.fromkeys(truth)}
    own: dict[str, str] = {}
    for p in sorted(per_person, key=lambda p: -sum(per_person[p].values())):
        free = [(n, l) for l, n in per_person[p].items() if l not in own.values()]
        if free:
            own[p] = max(free)[1]
    wrong = sum(1 for l, t in zip(labels, truth) if own.get(t) != l)
    return {p: len(c) for p, c in per_person.items()}, wrong / len(labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--truth", type=Path, help="TSV: file name, person")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.45])
    p.add_argument("--min-new", type=float, default=1.5, help="seconds of speech needed to start a speaker")
    args = p.parse_args()

    files = sorted(args.folder.glob("*-system.wav"))  # names start with the phrase start time
    if not files:
        sys.exit(f"No *-system.wav files in {args.folder}. Record with: python -m livetx --save-segments {args.folder}")
    truth = None
    if args.truth:
        rows = dict(line.split("\t") for line in args.truth.read_text(encoding="utf-8").splitlines() if line.strip())
        missing = [f.name for f in files if f.name not in rows]
        if missing:
            sys.exit(f"{args.truth} has no person for: {', '.join(missing)}")
        truth = [rows[f.name].strip() for f in files]

    embed, trim = ecapa_embedder(), silero_trimmer()
    audio = [read_wav(f) for f in files]
    speech = [trim(a) for a in audio]
    voiced = {id(a): s for a, s in zip(audio, speech)}  # trim once, reuse for every threshold
    cache: dict[bytes, object] = {}

    def cached_embed(x):
        key = x.tobytes()
        if key not in cache:
            cache[key] = embed(x)
        return cache[key]

    for threshold in args.threshold:
        tracker = SpeakerTracker(cached_embed, threshold, trim=lambda a: voiced[id(a)], min_new_s=args.min_new)
        labels, sims = [], []
        for f, a in zip(files, audio):
            labels.append(tracker.label(Segment("system", 0.0, a.size / 16_000, a)))
            sims.append(tracker.last_similarity)
        print(f"\nthreshold {threshold}, min new speaker {args.min_new} s speech")
        print(f"{'file':<24}{'length s':>9}{'speech s':>9}{'similarity':>11}  label        {'truth' if truth else ''}")
        for i, f in enumerate(files):
            sim = "" if sims[i] is None else f"{sims[i]:.2f}"
            print(f"{f.name:<24}{audio[i].size / 16_000:>9.2f}{speech[i].size / 16_000:>9.2f}{sim:>11}  "
                  f"{labels[i]:<12} {truth[i] if truth else ''}")
        if truth:
            per_person, wrong = score(labels, truth)
            print("labels per person: " + ", ".join(f"{p} {n}" for p, n in per_person.items())
                  + f" | wrong lines: {wrong:.0%}")


if __name__ == "__main__":
    main()
