"""Character-level forensics: invisible marks, homoglyphs, typographic polish.

WHY THIS WORKS
--------------
Every other text detector reasons about *words*. This one reasons about the
*code points*, which is a completely independent axis of evidence and survives
paraphrasing that defeats stylometry.

  * Invisible characters. Zero-width spaces/joiners, word joiners, soft hyphens
    and variation selectors are never typed by a human on a keyboard. They
    appear when (a) a provider embeds a steganographic watermark, or (b) an
    "AI humanizer" tool splices text to fool naive detectors. Either way their
    presence is a strong, hard-to-fake tell.
  * Homoglyph substitution. Evasion tools swap Latin ``a`` for Cyrillic ``а``
    (U+0430) so the string reads identically but tokenizes differently. A word
    that mixes scripts is essentially never produced by honest writing.
  * Typographic uniformity. LLM output is *consistently* typeset: curly quotes,
    real em dashes and ellipsis characters, exactly one space after a period,
    no stray double spaces, no mixed quote styles. Human keyboard text is
    typographically messy — it mixes ``"`` with ``“``, leaves double spaces,
    and uses ``--`` for a dash. Perfect polish with zero inconsistency is
    itself suspicious.

Each signal is bounded and the fused result is P(AI-generated) in [0, 1].
``unicode_forensics_report`` returns the raw evidence for explainability.
"""
from __future__ import annotations

import re
import unicodedata

# Code points that carry no visible glyph and are never typed deliberately.
_INVISIBLE = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner (legitimate inside emoji sequences)
    "⁠",  # word joiner
    "⁡", "⁢", "⁣", "⁤",  # invisible math operators
    "﻿",  # BOM used mid-string
    "­",  # soft hyphen
    "᠎",  # Mongolian vowel separator
}
_VARIATION_SELECTORS = range(0xFE00, 0xFE10)

# Non-breaking / exotic spaces: pasted-from-a-renderer or deliberately inserted.
_ODD_SPACES = {" ", " ", " ", " ", " ", " "}

# "Smart" typography an LLM emits uniformly.
_SMART = {"’", "‘", "“", "”", "—", "–", "…"}

_LATIN_LOOKALIKE_SCRIPTS = ("CYRILLIC", "GREEK", "CHEROKEE", "ARMENIAN")


def _script(ch: str) -> str:
    """Coarse script name for a letter ('LATIN', 'CYRILLIC', ...), else ''."""
    if not ch.isalpha():
        return ""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    return name.split(" ")[0]


def _is_emoji_zwj(text: str, i: int) -> bool:
    """True if the ZWJ at index ``i`` is a legitimate emoji joiner.

    Emoji ZWJ sequences (family, profession and flag emoji) are ordinary human
    text, so they must not be counted as a hidden watermark.
    """
    for j in (i - 1, i + 1):
        if 0 <= j < len(text) and ord(text[j]) >= 0x2190:
            return True
    return False


def unicode_forensics_report(text: str) -> dict:
    """Raw per-signal evidence behind :func:`unicode_forensics_score`."""
    n = len(text)
    invisible = 0
    for i, ch in enumerate(text):
        if ch == "‍" and _is_emoji_zwj(text, i):
            continue
        if ch in _INVISIBLE or ord(ch) in _VARIATION_SELECTORS:
            invisible += 1

    odd_spaces = sum(1 for ch in text if ch in _ODD_SPACES)

    # Words whose letters come from more than one script -> homoglyph swap.
    mixed_script_words = 0
    for word in re.findall(r"\w{2,}", text):
        scripts = {s for s in (_script(c) for c in word) if s}
        if "LATIN" in scripts and scripts & set(_LATIN_LOOKALIKE_SCRIPTS):
            mixed_script_words += 1

    smart = sum(1 for ch in text if ch in _SMART)
    straight_quotes = text.count('"') + text.count("'")
    double_spaces = len(re.findall(r"(?<=[.!?]) {2,}|\w  +\w", text))
    ascii_dashes = len(re.findall(r"\w--\w|\w -- \w", text))
    ascii_ellipsis = text.count("...")

    return {
        "chars": n,
        "invisible": invisible,
        "odd_spaces": odd_spaces,
        "mixed_script_words": mixed_script_words,
        "smart_punctuation": smart,
        "straight_quotes": straight_quotes,
        "double_spaces": double_spaces,
        "ascii_dashes": ascii_dashes,
        "ascii_ellipsis": ascii_ellipsis,
    }


def unicode_forensics_score(text: str) -> float:
    """Fuse the character-level signals into P(AI-generated) in [0, 1].

    Returns the neutral 0.5 for text too short to judge *unless* a hard signal
    (invisible characters, homoglyphs) is present — those need no sample size.
    """
    if not isinstance(text, str):
        raise TypeError("unicode_forensics_score expects str")

    r = unicode_forensics_report(text)
    n = max(r["chars"], 1)

    # 1. Invisible characters. One is odd; three or more is a deliberate mark.
    #    Diminishing-returns curve: 1 -> 0.60, 2 -> 0.84, 3+ -> 0.94.
    hidden_signal = 1.0 - 0.4 ** r["invisible"] if r["invisible"] else 0.0

    # 2. Homoglyph mixing. Even a single mixed-script word is a loud signal.
    homoglyph_signal = 1.0 - 0.35 ** r["mixed_script_words"] if r["mixed_script_words"] else 0.0

    hard = max(hidden_signal, homoglyph_signal)
    if n < 80 and hard == 0.0:
        return 0.5

    # 3. Typographic polish: smart punctuation present AND no ASCII messiness.
    smart_rate = r["smart_punctuation"] / n
    messy = r["straight_quotes"] + r["double_spaces"] + r["ascii_dashes"] + r["ascii_ellipsis"]
    polish = min(1.0, smart_rate / 0.006)          # ~6 marks per 1000 chars saturates
    polish *= 1.0 if messy == 0 else max(0.0, 1.0 - messy / 4.0)

    # 4. Exotic whitespace (rendered-then-copied or spliced text).
    space_signal = min(1.0, r["odd_spaces"] / 3.0)

    # Typography is weak evidence either way, so soft-only results stay inside
    # [0.30, 0.70]; hard evidence (hidden marks, homoglyphs) dominates when present.
    soft = 0.70 * polish + 0.30 * space_signal
    return float(min(1.0, hard + (1.0 - hard) * (0.30 + 0.40 * soft)))
