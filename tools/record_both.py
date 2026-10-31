"""Record the mic and the system audio at the same time into two WAV files.

Step 1 check: proves both sources can be captured together. Prints a level meter per
second so you can see that each side actually receives sound.

    python tools/record_both.py --seconds 10
    python tools/record_both.py --mic 3 --system alsa_output....monitor --out recordings/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetx import audio  # noqa: E402
from livetx.audio import write_wav  # noqa: E402


def drain(src: audio.AudioSource) -> np.ndarray:
    chunks = []
    while not src.blocks.empty():
        chunks.append(src.blocks.get())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mic", help="source index or name (default: system default input)")
    p.add_argument("--system", help="monitor source index or name (default: monitor of default output)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=Path("recordings"))
    args = p.parse_args()

    sources = audio.list_sources()
    mic = audio.resolve_source(args.mic, sources) if args.mic else audio.default_mic()
    system = audio.resolve_source(args.system, sources) if args.system else audio.default_monitor()
    print(f"mic:    {mic}\nsystem: {system}\nRecording {args.seconds:.0f} s. Speak, and play something through the speakers.\n")

    srcs = {"mic": audio.AudioSource(mic, "mic"), "system": audio.AudioSource(system, "system")}
    collected = {k: [] for k in srcs}
    for s in srcs.values():
        s.start()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(1.0)
            line = []
            for k, s in srcs.items():
                chunk = drain(s)
                collected[k].append(chunk)
                rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
                line.append(f"{k}: {'#' * min(40, int(rms * 400)):<40}")
            print("  ".join(line))
    finally:
        for s in srcs.values():
            s.stop()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for k in srcs:
        samples = np.concatenate(collected[k] + [drain(srcs[k])])
        path = args.out / f"{stamp}-{k}.wav"
        write_wav(path, samples)
        print(f"wrote {path}  ({samples.size / audio.SAMPLE_RATE:.1f} s, peak {np.abs(samples).max(initial=0):.2f})")


if __name__ == "__main__":
    main()
