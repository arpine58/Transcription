"""From the brief: mic is always "Me"; call-side people get Speaker 1, 2, ... and keep their label."""
import numpy as np

from livetx.segmenter import Segment
from livetx.speakers import SpeakerTracker

rng = np.random.default_rng(0)
VOICE_A, VOICE_B = rng.normal(size=192), rng.normal(size=192)


def seg(source, seconds=2.0):
    return Segment(source, 0.0, seconds, np.zeros(int(16_000 * seconds), dtype=np.float32))


def _noisy(v):
    e = v + rng.normal(scale=0.3, size=v.size)
    return e / np.linalg.norm(e)


def test_mic_is_always_me():
    t = SpeakerTracker(embed=lambda a: (_ for _ in ()).throw(AssertionError("no embedding for mic")))
    assert t.label(seg("mic")) == "Me"


def test_two_voices_keep_their_labels():
    t = SpeakerTracker(embed=lambda a: None, threshold=0.5)
    order = [VOICE_A, VOICE_B, VOICE_A, VOICE_A, VOICE_B, VOICE_A, VOICE_B]
    labels = [t.assign(_noisy(v)) for v in order]
    assert labels == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"]


def test_short_segment_reuses_last_call_speaker():
    t = SpeakerTracker(embed=lambda a: _noisy(VOICE_B), threshold=0.5)
    assert t.label(seg("system", seconds=2.0)) == "Speaker 1"
    t.embed = lambda a: (_ for _ in ()).throw(AssertionError("too short to embed"))
    assert t.label(seg("system", seconds=0.6)) == "Speaker 1"


def test_short_phrase_cannot_start_a_new_speaker():
    t = SpeakerTracker(embed=lambda a: _noisy(VOICE_A), threshold=0.5, min_new_s=2.0)
    assert t.label(seg("system", seconds=3.0)) == "Speaker 1"
    t.embed = lambda a: _noisy(VOICE_B)
    assert t.label(seg("system", seconds=1.5)) == "Speaker 1"  # too short to start a speaker
    assert t.label(seg("system", seconds=3.0)) == "Speaker 2"


def test_only_voiced_audio_counts():
    # a 2.5 s phrase with 0.8 s of speech is treated as short: no embedding, previous speaker
    t = SpeakerTracker(embed=lambda a: _noisy(VOICE_A), threshold=0.5, trim=lambda a: a[: int(16_000 * 0.8)])
    t.last_call_speaker = "Speaker 1"
    t.embed = lambda a: (_ for _ in ()).throw(AssertionError("too little speech to embed"))
    assert t.label(seg("system", seconds=2.5)) == "Speaker 1"
    assert t.last_speech_s == 0.8


def test_two_speakers_that_turn_out_alike_are_merged():
    t = SpeakerTracker(embed=lambda a: None, threshold=0.5)
    mid = (VOICE_A + VOICE_B) / np.linalg.norm(VOICE_A + VOICE_B)
    assert t.assign(VOICE_A / np.linalg.norm(VOICE_A)) == "Speaker 1"
    assert t.assign(VOICE_B / np.linalg.norm(VOICE_B)) == "Speaker 2"
    # a run of phrases between the two pulls Speaker 2 towards Speaker 1 until they merge
    labels = [t.assign(mid) for _ in range(5)]
    assert labels[-1] == "Speaker 1"
    assert sum(c is not None for c in t.centroids) == 1
