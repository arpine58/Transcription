"""From the 5 s requirement: a backlog is caught up by joining waiting phrases, never by dropping them."""
import numpy as np

from livetx.pipeline import Pipeline
from livetx.segmenter import Segment


def seg(source, start, seconds):
    s = Segment(source, start, start + seconds, np.zeros(int(16_000 * seconds), dtype=np.float32))
    s.timings["speech_end"] = start + seconds
    return s


def pipeline(merge_s=15.0):
    return Pipeline([], transcriber=None, translator=None, speakers=None, latency_log=None, merge_s=merge_s)


def test_single_phrase_is_not_changed():
    p = pipeline()
    s = seg("mic", 0.0, 2.0)
    p.segments.put(s)
    assert p._next_batch() == [s]


def test_waiting_phrases_are_joined_per_source_in_time_order():
    p = pipeline()
    for s in [seg("system", 0.0, 2.0), seg("mic", 1.0, 1.0), seg("system", 3.0, 2.0), seg("mic", 4.0, 1.0)]:
        p.segments.put(s)
    batch = p._next_batch()
    assert [(b.source, b.start, b.audio.size / 16_000) for b in batch] == [("system", 0.0, 4.0), ("mic", 1.0, 2.0)]
    assert batch[0].timings["speech_end"] == 2.0  # latency counts from the first phrase


def test_joined_phrases_stay_under_the_limit():
    p = pipeline(merge_s=5.0)
    for i in range(4):
        p.segments.put(seg("system", 2.0 * i, 2.0))
    assert [b.audio.size / 16_000 for b in p._next_batch()] == [4.0, 4.0]
