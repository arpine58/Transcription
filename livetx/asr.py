"""Armenian speech -> Armenian text with Whisper.

Two backends, picked by the model spec:
  <name or path>   faster-whisper (CTranslate2, int8 on CPU)
  ov:<dir>[@DEV]   OpenVINO GenAI on the Intel GPU (default) or NPU/CPU, model exported with optimum-cli

In: 16 kHz float32 audio. Out: a Transcript with the text and Whisper's own confidence signals,
so the pipeline can show unreliable output as unclear instead of printing hallucinations.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

import numpy as np

_ARMENIAN = re.compile(r"[Ա-֏]")
_LETTER = re.compile(r"[^\W\d_]")

# Whisper's own defaults for detecting a failed decode (openai/whisper transcribe.py)
MAX_COMPRESSION_RATIO = 2.4  # higher = repetitive output like "өk өk өk өk"
MIN_AVG_LOGPROB = -1.0
MIN_ARMENIAN_RATIO = 0.8  # share of letters in the Armenian alphabet


def armenian_ratio(text: str) -> float:
    letters = _LETTER.findall(text)
    return sum(1 for ch in letters if _ARMENIAN.match(ch)) / len(letters) if letters else 0.0


@dataclass
class Transcript:
    text: str
    avg_logprob: float | None  # None when the backend does not report it (OpenVINO)
    compression_ratio: float

    def problem(self) -> str | None:
        """Why this transcript should not be trusted, or None if it looks fine."""
        if not self.text:
            return "no speech recognised"
        if armenian_ratio(self.text) < MIN_ARMENIAN_RATIO:
            return "not Armenian script"
        if self.compression_ratio > MAX_COMPRESSION_RATIO:
            return "repetitive output"
        if self.avg_logprob is not None and self.avg_logprob < MIN_AVG_LOGPROB:
            return "low confidence"
        return None


def compression_ratio(text: str) -> float:
    """Same definition as Whisper: raw bytes / zlib-compressed bytes. Repetitive text compresses well."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def make_transcriber(spec: str, threads: int = 8):
    if spec.startswith("ov:"):
        path, _, device = spec[3:].partition("@")
        return OpenVINOTranscriber(path, device or "GPU")
    return Transcriber(spec, threads=threads)


class OpenVINOTranscriber:
    def __init__(self, model_dir: str, device: str = "GPU"):
        import openvino_genai

        self.pipe = openvino_genai.WhisperPipeline(model_dir, device)
        self.transcribe(np.zeros(16_000, dtype=np.float32))  # warm-up (GPU kernel compilation)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        # Cap the output length so a looping decode ("Մանդրանդրանդր...") stops early instead of taking 25 s.
        # Armenian needs many tokens per word; 25 tokens per second of speech is well above normal speech.
        max_tokens = int(10 + 25 * audio.size / 16_000)
        result = self.pipe.generate(audio.tolist(), max_new_tokens=max_tokens, language="<|hy|>", task="transcribe")
        text = result.texts[0].strip()
        return Transcript(text=text, avg_logprob=None, compression_ratio=compression_ratio(text))


class Transcriber:
    def __init__(self, model: str = "large-v3", threads: int = 8, beam_size: int = 1):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        self.beam_size = beam_size
        # The first call is much slower (memory allocation, kernel setup); pay it at startup, not on the first phrase.
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> Transcript:
        segments, _ = self.model.transcribe(
            audio,
            language="hy",
            beam_size=self.beam_size,
            condition_on_previous_text=False,  # each phrase on its own; avoids repeated hallucinations
            vad_filter=False,  # we already cut segments with our own VAD
        )
        segments = list(segments)
        if not segments:
            return Transcript("", 0.0, 0.0)
        return Transcript(
            text=" ".join(s.text.strip() for s in segments).strip(),
            avg_logprob=min(s.avg_logprob for s in segments),
            compression_ratio=max(s.compression_ratio for s in segments),
        )
