from bench.textnorm import normalize


def test_strips_armenian_punctuation_and_case():
    assert normalize("Բարև, ինչպե՞ս ես։") == "բարև ինչպես ես"


def test_collapses_whitespace():
    assert normalize("  Լավ   եմ ․ ") == "լավ եմ"
