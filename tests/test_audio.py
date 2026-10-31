import pytest

from livetx.audio import parse_short_sources, resolve_source

PACTL = (
    "51\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n"
    "52\talsa_input.pci-0000_00_1f.3.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"
)


def test_parse_marks_monitor_sources():
    sources = parse_short_sources(PACTL)
    assert [s.index for s in sources] == [51, 52]
    assert sources[0].is_monitor and not sources[1].is_monitor


def test_resolve_by_index_and_name():
    sources = parse_short_sources(PACTL)
    assert resolve_source("52", sources) == "alsa_input.pci-0000_00_1f.3.analog-stereo"
    assert resolve_source(sources[0].name, sources) == sources[0].name


def test_resolve_unknown_fails_loudly():
    with pytest.raises(ValueError, match="Unknown audio source"):
        resolve_source("nope", parse_short_sources(PACTL))
