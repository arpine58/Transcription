"""Hallucinated output must be flagged, never printed as a translation (from Arpine's first live test)."""
import sys
import types

# asr.py imports faster_whisper at module level; the checks below don't need the model itself
sys.modules.setdefault("faster_whisper", types.SimpleNamespace(WhisperModel=None))

from livetx.asr import Transcript, armenian_ratio, compression_ratio  # noqa: E402


def test_armenian_ratio():
    assert armenian_ratio("Ես խոսում եմ հայերեն։") == 1.0
    assert armenian_ratio("You won't need a fixed track.") == 0.0


def test_real_hallucinations_from_first_run_are_flagged():
    for text in ["— You won't need a fixed track.", "ད ད ད", "ཡվ ཡվ སྯ Սա ག འག ག ག", "�aut— ә өk өk өk өk"]:
        assert Transcript(text, -0.3, 1.5).problem() == "not Armenian script"


def test_repetition_and_low_confidence_are_flagged():
    assert Transcript("ես ես ես ես ես ես ես ես", -0.3, 3.0).problem() == "repetitive output"
    assert Transcript("Բարև", -1.5, 1.0).problem() == "low confidence"


def test_good_armenian_passes():
    t = Transcript("Հագինքվեսը պլատվորմա որտեղ որ մարդիկ պաբլիշ են անում իրանց մոդելները։", -0.4, 1.3)
    assert t.problem() is None


def test_compression_ratio_flags_repetition():
    assert compression_ratio("ես " * 30) > 2.4
    assert compression_ratio("Ես խոսում եմ հայերեն։") < 2.4


def test_missing_logprob_is_not_a_problem():
    assert Transcript("Ես խոսում եմ հայերեն։", None, 1.0).problem() is None
