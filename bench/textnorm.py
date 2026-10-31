"""Text normalisation for Armenian WER: lowercase, drop punctuation, collapse spaces."""
import re

# ՞ (question) ՜ (exclamation) ՛ (emphasis) sit inside a word, so they are deleted.
_IN_WORD = re.compile(r"[՞՜՛]")
# ։ (full stop) ՝ (comma) ․ (dot), quotes and ASCII punctuation separate words.
_PUNCT = re.compile(r"[։՝․«»\"'.,!?;:()\-–—…]")


def normalize(text: str) -> str:
    text = _PUNCT.sub(" ", _IN_WORD.sub("", text.lower()))
    return " ".join(text.split())
