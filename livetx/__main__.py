"""Live Armenian -> English transcription. `python -m livetx --help`."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from livetx import audio


def list_devices() -> None:
    print(f"Default mic:            {audio.default_mic()}")
    print(f"Default system monitor: {audio.default_monitor()}\n")
    for s in audio.list_sources():
        kind = "system audio (monitor)" if s.is_monitor else "input"
        print(f"{s.index:>4}  {s.name}  [{kind}]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="livetx", description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="print audio sources and exit")
    parser.add_argument("--mic", help="mic source index or name (default: system default input)")
    parser.add_argument("--system", help="monitor source index or name (default: monitor of default output)")
    parser.add_argument("--asr-model", default="ov:models/whisper-large-v3-int4-ov@GPU",
                        help="faster-whisper name/path, or ov:<dir>[@GPU|NPU|CPU] for OpenVINO")
    parser.add_argument("--nllb-dir", type=Path, default=Path("models/nllb-200-distilled-600M-ct2"))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--latency-log", type=Path, default=Path("latency.tsv"),
                        help="per-segment latency log (tab separated)")
    parser.add_argument("--min-silence", type=float, default=0.4,
                        help="seconds of silence that end a phrase")
    parser.add_argument("--max-segment", type=float, default=6.0,
                        help="seconds; a phrase longer than this is cut and printed in parts")
    parser.add_argument("--speaker-threshold", type=float, default=0.45,
                        help="cosine similarity needed to count as the same call-side speaker")
    parser.add_argument("--no-speakers", action="store_true", help="label the call side just 'Call'")
    parser.add_argument("--save-segments", type=Path, help="debug: save every segment as WAV in this folder")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # heavy imports only when actually running
    from livetx.asr import make_transcriber
    from livetx.pipeline import Pipeline
    from livetx.speakers import SourceLabels, SpeakerTracker, ecapa_embedder, silero_trimmer
    from livetx.translate import Translator

    sources = audio.list_sources()
    mic = audio.resolve_source(args.mic, sources) if args.mic else audio.default_mic()
    system = audio.resolve_source(args.system, sources) if args.system else audio.default_monitor()

    print(f"mic:    {mic}\nsystem: {system}\nloading models...", flush=True)
    pipeline = Pipeline(
        sources=[audio.AudioSource(mic, "mic"), audio.AudioSource(system, "system")],
        transcriber=make_transcriber(args.asr_model, threads=args.threads),
        translator=Translator(args.nllb_dir, threads=max(1, args.threads // 2)),
        speakers=SourceLabels() if args.no_speakers else SpeakerTracker(ecapa_embedder(), args.speaker_threshold,
                                                                                 trim=silero_trimmer()),
        latency_log=args.latency_log,
        save_segments=args.save_segments,
        max_segment_s=args.max_segment,
        min_silence_s=args.min_silence,
    )
    print("listening. Ctrl+C to stop.\n", flush=True)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
