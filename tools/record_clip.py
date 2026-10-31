"""Record one evaluation clip from the mic and write its reference text to eval/refs.tsv.

    python tools/record_clip.py clip01 --speaker arpine
    -> press Enter to start, say the phrase, press Enter to stop, then type exactly what you said.

Audio goes to eval/clips/<name>.wav (ignored by git). Only the text goes into git.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from livetx import audio  # noqa: E402
from tools.record_both import drain, write_wav  # noqa: E402

REFS = ROOT / "eval" / "refs.tsv"
CLIPS = ROOT / "eval" / "clips"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="clip id, e.g. clip01")
    p.add_argument("--speaker", required=True, help="who speaks, e.g. arpine")
    p.add_argument("--source", help="source index or name (default: default mic)")
    args = p.parse_args()

    if any(line.split("\t")[0] == f"{args.name}.wav" for line in REFS.read_text(encoding="utf-8").splitlines()):
        sys.exit(f"{args.name} already exists in {REFS}. Pick another name.")

    source = audio.resolve_source(args.source, audio.list_sources()) if args.source else audio.default_mic()
    input(f"Source: {source}\nPress Enter, then speak...")
    src = audio.AudioSource(source, "mic")
    src.start()
    input("Recording. Press Enter to stop.")
    src.stop()
    samples = drain(src)

    CLIPS.mkdir(parents=True, exist_ok=True)
    path = CLIPS / f"{args.name}.wav"
    write_wav(path, samples)
    print(f"wrote {path} ({samples.size / audio.SAMPLE_RATE:.1f} s, peak {np.abs(samples).max(initial=0):.2f})")

    text = input("Type exactly what was said (Armenian): ").strip()
    if not text:
        sys.exit("No text given; clip kept but not added to refs.tsv.")
    with REFS.open("a", encoding="utf-8") as f:
        f.write(f"{args.name}.wav\t{args.speaker}\t{text}\n")
    print(f"added to {REFS}")


if __name__ == "__main__":
    main()
