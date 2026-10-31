"""Wire the modules together.

  capture thread (per source) -> segmenter thread (per source) -> one queue -> ASR worker -> post worker
  ASR worker (GPU): Whisper, one phrase at a time
  post worker (CPU): speaker label -> translation -> print, while the GPU already works on the next phrase

One ASR worker, because there is one GPU; two in parallel would just slow each other down.
When phrases pile up, the ASR worker joins the waiting phrases of each source into longer ones: on the iGPU a
Whisper call costs about 2 s plus 0.5 s per second of audio (it always encodes a 30 s window), so fewer, longer
calls catch up with the speakers. Without a backlog every phrase is transcribed on its own.
Latency (end of phrase -> printed) is measured per line and written to the latency log.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

import numpy as np

from livetx.audio import BLOCK_SAMPLES, SAMPLE_RATE, AudioSource, write_wav
from livetx.output import format_line, format_unclear
from livetx.segmenter import Segment, Segmenter

if TYPE_CHECKING:  # type hints only, so the queue logic can be tested without the model libraries
    from livetx.asr import Transcriber, Transcript
    from livetx.speakers import SourceLabels, SpeakerTracker
    from livetx.translate import Translator


LATENCY_HEADER = ("start\tspeaker\tlength_s\tspeech_s\tparts\tqueue_s\tasr_s\tspeaker_s\tmt_s\ttotal_s"
                  "\tsimilarity\tresult\n")


class Pipeline:
    def __init__(self, sources: list[AudioSource], transcriber: Transcriber, translator: Translator,
                 speakers: SpeakerTracker | SourceLabels, latency_log: Path | None, save_segments: Path | None = None,
                 max_segment_s: float = 6.0, min_silence_s: float = 0.4, merge_s: float = 15.0):
        self.sources = sources
        self.transcriber = transcriber
        self.translator = translator
        self.speakers = speakers
        self.segments: Queue[Segment] = Queue()
        self.transcribed: Queue[tuple[Segment, Transcript]] = Queue()
        self.merge_s = merge_s
        self.latency_log = None
        if latency_log:
            new = not latency_log.exists() or latency_log.stat().st_size == 0
            self.latency_log = latency_log.open("a", encoding="utf-8")
            if new:
                self.latency_log.write(LATENCY_HEADER)
        self.t0 = 0.0
        self.max_segment_s = max_segment_s
        self.min_silence_s = min_silence_s
        self.save_segments = save_segments  # debug: write each segment as WAV to hear what ASR got
        if save_segments:
            save_segments.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self.t0 = time.monotonic()
        threading.excepthook = _exit_on_thread_error
        threading.Thread(target=self._post_loop, daemon=True).start()
        for src in self.sources:
            src.start()
            threading.Thread(target=self._segment_loop, args=(src,), daemon=True).start()
        self._worker_loop()  # main thread; Ctrl+C stops it

    def stop(self) -> None:
        for src in self.sources:
            src.stop()
        if self.latency_log:
            self.latency_log.close()

    def _segment_loop(self, src: AudioSource) -> None:
        seg = Segmenter(src.label, max_segment_s=self.max_segment_s, min_silence_s=self.min_silence_s)
        while True:
            block = src.blocks.get()
            for s in seg.feed(block):
                # time the phrase ended = now minus the audio still waiting in the queue
                s.timings["speech_end"] = time.monotonic() - src.blocks.qsize() * BLOCK_SAMPLES / SAMPLE_RATE
                self.segments.put(s)

    def _worker_loop(self) -> None:
        while True:
            for s in self._next_batch():
                s.timings["dequeued"] = time.monotonic()
                behind = s.timings["dequeued"] - s.timings["speech_end"]
                if behind > 5.0:
                    # not hidden: ASR is slower than speech and the queue is growing
                    print(f"  [behind by {behind:.0f} s, {self.segments.qsize()} segments waiting]", file=sys.stderr, flush=True)
                if self.save_segments:
                    write_wav(self.save_segments / f"{s.start:08.2f}-{s.source}.wav", s.audio)
                tr = self.transcriber.transcribe(s.audio)
                s.timings["asr"] = time.monotonic()
                self.transcribed.put((s, tr))

    def _next_batch(self) -> list[Segment]:
        """Wait for a phrase, then take everything waiting, each source's phrases joined up to merge_s."""
        waiting = [self.segments.get()]
        while True:
            try:
                waiting.append(self.segments.get_nowait())
            except Empty:
                break
        batch = []
        for source in dict.fromkeys(s.source for s in waiting):
            current = None
            for s in (s for s in waiting if s.source == source):
                if current is not None and (current.audio.size + s.audio.size) / SAMPLE_RATE <= self.merge_s:
                    current = _join(current, s)
                else:
                    if current is not None:
                        batch.append(current)
                    current = s
            batch.append(current)
        return sorted(batch, key=lambda s: s.start)

    def _post_loop(self) -> None:
        while True:
            s, tr = self.transcribed.get()
            s.timings["post"] = time.monotonic()
            speaker = self.speakers.label(s)
            s.timings["speaker"] = time.monotonic()
            problem = tr.problem()
            if problem:
                # shown, not hidden: the user sees that something was said but not understood
                print(format_unclear(s.start, speaker, problem, tr.text), flush=True)
            else:
                en = self.translator.translate(tr.text)
                print(format_line(s.start, speaker, tr.text, en), flush=True)
            s.timings["printed"] = time.monotonic()
            self._log_latency(s, speaker, f"unclear: {problem}" if problem else "ok")

    def _log_latency(self, s: Segment, speaker: str, result: str) -> None:
        if not self.latency_log:
            return
        t = s.timings
        call = s.source == "system"
        speech = getattr(self.speakers, "last_speech_s", None) if call else None
        similarity = getattr(self.speakers, "last_similarity", None) if call else None
        self.latency_log.write(
            f"{s.start:.2f}\t{speaker}\t{s.audio.size / SAMPLE_RATE:.2f}\t{_num(speech)}\t{s.parts}\t"
            f"{t['dequeued'] - t['speech_end']:.2f}\t{t['asr'] - t['dequeued']:.2f}\t"
            f"{t['speaker'] - t['post']:.2f}\t{t['printed'] - t['speaker']:.2f}\t{t['printed'] - t['speech_end']:.2f}\t"
            f"{_num(similarity)}\t{result}\n"
        )
        self.latency_log.flush()


def _num(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}"


def _join(a: Segment, b: Segment) -> Segment:
    """One segment from two consecutive phrases of the same source. Latency counts from the first phrase's end."""
    return Segment(a.source, a.start, b.end, np.concatenate([a.audio, b.audio]), dict(a.timings), a.parts + b.parts)


def _exit_on_thread_error(args: threading.ExceptHookArgs) -> None:
    """A crash in any thread (e.g. an audio source that stopped) ends the program instead of leaving it half alive."""
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    os._exit(1)
