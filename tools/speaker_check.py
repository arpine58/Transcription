"""Tune the speaker threshold: print cosine similarity between ECAPA embeddings of WAV files.

    python tools/speaker_check.py eval/clips/clip01.wav eval/clips/clip02.wav segs/*-system.wav

Same person should score clearly higher than different people. Pick a threshold between the two groups
and pass it with `python -m livetx --speaker-threshold X`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetx.audio import read_wav  # noqa: E402
from livetx.speakers import ecapa_embedder  # noqa: E402


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) < 2:
        sys.exit(__doc__)
    embed = ecapa_embedder()
    embs = [embed(read_wav(p)) for p in paths]
    embs = [e / np.linalg.norm(e) for e in embs]
    names = [p.name[:18] for p in paths]
    print(" " * 20 + " ".join(f"{n:>18}" for n in names))
    for n, a in zip(names, embs):
        print(f"{n:<20}" + " ".join(f"{float(a @ b):>18.2f}" for b in embs))


if __name__ == "__main__":
    main()
