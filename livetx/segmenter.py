"""Split a stream of audio blocks into speech segments with Silero VAD.

In:  30 ms float32 blocks at 16 kHz from one source.
Out: Segment objects, emitted when a phrase ends (silence) or hits the max length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livetx.audio import SAMPLE_RATE

VAD_CHUNK = 512  # Silero needs exactly 512 samples per call at 16 kHz


@dataclass
class Segment:
    source: str  # "mic" or "system"
    start: float  # seconds since session start
    end: float
    audio: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)  # wall-clock stage timestamps
    parts: int = 1  # phrases joined into this segment when the queue was behind


class Segmenter:
    def __init__(
        self,
        source: str,
        threshold: float = 0.5,
        min_silence_s: float = 0.4,
        max_segment_s: float = 6.0,
        min_speech_s: float = 0.5,
        pre_roll_s: float = 0.2,
    ):
        import torch  # heavy; imported here so Segment can be used without it (tests, speakers)
        from silero_vad import load_silero_vad

        self._torch = torch
        self.source = source
        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_silence = int(min_silence_s * SAMPLE_RATE)
        self.max_segment = int(max_segment_s * SAMPLE_RATE)
        self.min_speech = int(min_speech_s * SAMPLE_RATE)
        self.pre_roll = int(pre_roll_s * SAMPLE_RATE)
        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet fed to the VAD
        self._history = np.zeros(0, dtype=np.float32)  # recent non-speech audio, for pre-roll
        self._speech: list[np.ndarray] = []
        self._speech_len = 0
        self._silence_len = 0
        self._samples_seen = 0
        self._speech_start = 0

    def feed(self, block: np.ndarray) -> list[Segment]:
        self._pending = np.concatenate([self._pending, block])
        out = []
        while self._pending.size >= VAD_CHUNK:
            chunk, self._pending = self._pending[:VAD_CHUNK], self._pending[VAD_CHUNK:]
            seg = self._step(chunk)
            if seg is not None:
                out.append(seg)
        return out

    def _step(self, chunk: np.ndarray) -> Segment | None:
        prob = self.model(self._torch.from_numpy(chunk), SAMPLE_RATE).item()
        self._samples_seen += chunk.size
        is_speech = prob >= self.threshold

        if not self._speech:
            if is_speech:
                self._speech = [self._history, chunk]
                self._speech_len = self._history.size + chunk.size
                self._speech_start = self._samples_seen - self._speech_len
                self._silence_len = 0
            else:
                self._history = np.concatenate([self._history, chunk])[-self.pre_roll:]
            return None

        self._speech.append(chunk)
        self._speech_len += chunk.size
        self._silence_len = 0 if is_speech else self._silence_len + chunk.size
        if self._silence_len >= self.min_silence or self._speech_len >= self.max_segment:
            return self._close()
        return None

    def _close(self) -> Segment | None:
        audio = np.concatenate(self._speech)
        voiced = self._speech_len - self._silence_len
        self._speech, self._speech_len, self._silence_len = [], 0, 0
        self._history = np.zeros(0, dtype=np.float32)
        self.model.reset_states()
        if voiced < self.min_speech:
            return None  # a click or a breath, not a phrase
        start = self._speech_start / SAMPLE_RATE
        return Segment(self.source, start, start + audio.size / SAMPLE_RATE, audio)
