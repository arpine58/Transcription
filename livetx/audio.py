"""Audio capture from PipeWire/PulseAudio sources.

Each source (the microphone, or the `.monitor` of the speaker output for call audio)
is read by its own `parec` process as 16 kHz mono 16-bit PCM. We use `parec` instead of
PortAudio because PortAudio on Linux only sees ALSA devices and cannot pick a monitor
source by name. See DECISIONS.md.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass
from queue import Queue

import numpy as np

SAMPLE_RATE = 16_000
BLOCK_MS = 30
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
_BYTES_PER_SAMPLE = 2  # s16le


@dataclass(frozen=True)
class SourceInfo:
    index: int
    name: str
    is_monitor: bool


def _run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(
            f"'{cmd[0]}' not found. Install it with: sudo apt install pulseaudio-utils"
        )
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_short_sources(text: str) -> list[SourceInfo]:
    """Parse the output of `pactl list short sources` (tab separated)."""
    sources = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        name = fields[1]
        sources.append(SourceInfo(int(fields[0]), name, name.endswith(".monitor")))
    return sources


def list_sources() -> list[SourceInfo]:
    return parse_short_sources(_run(["pactl", "list", "short", "sources"]))


def default_mic() -> str:
    return _run(["pactl", "get-default-source"]).strip()


def default_monitor() -> str:
    return _run(["pactl", "get-default-sink"]).strip() + ".monitor"


def write_wav(path, samples: np.ndarray) -> None:
    """Write float32 samples as 16 kHz mono 16-bit WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(_BYTES_PER_SAMPLE)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit WAV as float32 samples."""
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != _BYTES_PER_SAMPLE:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit, got {w.getframerate()} Hz {w.getnchannels()} ch")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def resolve_source(choice: str, sources: list[SourceInfo]) -> str:
    """Accept a source index or exact name, return the name. Fail loudly otherwise."""
    for s in sources:
        if choice == s.name or choice == str(s.index):
            return s.name
    names = "\n  ".join(f"{s.index}: {s.name}" for s in sources)
    raise ValueError(f"Unknown audio source '{choice}'. Available:\n  {names}")


class AudioSource:
    """Reads one PulseAudio/PipeWire source and puts float32 blocks on a queue."""

    def __init__(self, source_name: str, label: str):
        self.source_name = source_name
        self.label = label  # "mic" or "system"
        self.blocks: Queue[np.ndarray] = Queue()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        if shutil.which("parec") is None:
            raise RuntimeError("'parec' not found. Install it with: sudo apt install pulseaudio-utils")
        self._proc = subprocess.Popen(
            [
                "parec",
                f"--device={self.source_name}",
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--raw",
                f"--latency-msec={BLOCK_MS}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        nbytes = BLOCK_SAMPLES * _BYTES_PER_SAMPLE
        while True:
            data = self._proc.stdout.read(nbytes)
            if not data:
                break
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            self.blocks.put(samples)
        if self._stopping:
            return
        # The stream ended without stop() being called: fail loudly instead of going silent.
        code = self._proc.wait()
        err = self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
        raise RuntimeError(f"Audio source '{self.source_name}' stopped unexpectedly (parec exit {code}): {err or 'no error output'}")

    def stop(self) -> None:
        self._stopping = True
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
