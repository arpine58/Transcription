"""Terminal formatting of one transcribed line."""
from __future__ import annotations


def _ts(start_s: float) -> str:
    h, rem = divmod(int(start_s), 3600)
    m, s = divmod(rem, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def format_unclear(start_s: float, speaker: str, reason: str, raw: str) -> str:
    return f"{_ts(start_s)} {speaker:<10} (unclear: {reason}) raw: {raw[:60]}"


def format_line(start_s: float, speaker: str, hy: str, en: str) -> str:
    return f"{_ts(start_s)} {speaker:<10} HY: {hy}\n{'':22}EN: {en}"
