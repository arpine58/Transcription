"""Speaker labels.

Mic side: always "Me".
Call side: a speaker embedding per segment (SpeechBrain ECAPA-TDNN, 192 dims), then simple online clustering.
Each known speaker keeps a running mean embedding. A segment joins the most similar speaker if the cosine
similarity is above the threshold, otherwise it starts "Speaker N+1". Labels never change once given, so the
same person keeps the same label for the session. The number of speakers is not limited.

On Oct 30 one remote person became 8 speakers: the new speakers came from short phrases, whose embeddings
scatter (same-person similarity 0.22-0.68 over Zoom). So:
- only the voiced part of a phrase is embedded (no pre-roll, no trailing silence);
- under `min_embed_s` of speech there is no embedding: the phrase gets the previous call speaker;
- only `min_new_s` of speech or more can start a new speaker; shorter phrases join the closest one;
- only a confident match (above the threshold) updates a speaker's voice;
- two speakers whose voices end up closer than the threshold are merged into the older one.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from livetx.audio import SAMPLE_RATE
from livetx.segmenter import Segment

Embedder = Callable[[np.ndarray], np.ndarray]
Trimmer = Callable[[np.ndarray], np.ndarray]  # audio -> only its voiced parts


def ecapa_embedder(savedir: str = "models/spkrec-ecapa-voxceleb") -> Embedder:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
                                           run_opts={"device": "cpu"})

    def embed(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model.encode_batch(torch.from_numpy(audio)[None]).squeeze().numpy()

    return embed


def silero_trimmer() -> Trimmer:
    """Keep only the speech in a phrase, using the same Silero VAD that cuts the phrases."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()

    def trim(audio: np.ndarray) -> np.ndarray:
        stamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SAMPLE_RATE)
        if not stamps:
            return audio[:0]
        return np.concatenate([audio[s["start"]:s["end"]] for s in stamps])

    return trim


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class SpeakerTracker:
    def __init__(self, embed: Embedder, threshold: float = 0.45, trim: Trimmer = lambda a: a,
                 min_embed_s: float = 1.0, min_new_s: float = 1.5):
        self.embed = embed
        self.trim = trim
        self.threshold = threshold
        self.min_embed = int(min_embed_s * SAMPLE_RATE)
        self.min_new = int(min_new_s * SAMPLE_RATE)
        self.centroids: list[np.ndarray | None] = []  # unit-length running means, index i = "Speaker i+1"
        self.counts: list[int] = []
        self.last_call_speaker: str | None = None
        self.last_similarity: float | None = None  # for the latency/debug log
        self.last_speech_s: float | None = None  # voiced seconds in the last call phrase, for the log

    def label(self, segment: Segment) -> str:
        if segment.source == "mic":
            return "Me"
        speech = self.trim(segment.audio)
        self.last_speech_s = speech.size / SAMPLE_RATE
        if speech.size < self.min_embed and self.last_call_speaker is not None:
            # too little speech for a reliable embedding: most likely the person who was just talking
            self.last_similarity = None
            return self.last_call_speaker
        if speech.size < self.min_embed:
            speech = segment.audio  # the very first call phrase is short: embed what there is
        self.last_call_speaker = self.assign(_unit(self.embed(speech)), may_start_new=speech.size >= self.min_new)
        return self.last_call_speaker

    def assign(self, e: np.ndarray, may_start_new: bool = True) -> str:
        sims = {i: float(c @ e) for i, c in enumerate(self.centroids) if c is not None}
        best = max(sims, key=sims.get) if sims else -1
        self.last_similarity = sims[best] if sims else None
        if best < 0 or (sims[best] < self.threshold and may_start_new):
            self.centroids.append(e)
            self.counts.append(1)
            return f"Speaker {len(self.centroids)}"
        if sims[best] >= self.threshold:
            # only a confident match updates the voice; a forced one (a short phrase) would blur it
            n = self.counts[best]
            self.centroids[best] = _unit(self.centroids[best] * n + e)
            self.counts[best] = n + 1
            best = self._merge_close(best)
        return f"Speaker {best + 1}"

    def _merge_close(self, i: int) -> int:
        """If speaker i now sounds like another speaker, fold the newer one into the older. Returns i's new index."""
        for j, c in enumerate(self.centroids):
            if j == i or c is None or self.centroids[i] is None or float(c @ self.centroids[i]) < self.threshold:
                continue
            keep, drop = min(i, j), max(i, j)
            n, m = self.counts[keep], self.counts[drop]
            self.centroids[keep] = _unit(self.centroids[keep] * n + self.centroids[drop] * m)
            self.counts[keep] = n + m
            self.centroids[drop] = None  # its label is never given again
            i = keep
        return i


class SourceLabels:
    """Labels without a speaker model: mic -> "Me", system -> "Call"."""

    def label(self, segment: Segment) -> str:
        return "Me" if segment.source == "mic" else "Call"
