"""Script-aware tokenisation shared by the model-free text detectors.

WHY THIS EXISTS
---------------
The original detectors tokenised with ``[a-zA-Z']+``, which returns *zero*
tokens for Chinese, Japanese, Korean, Arabic, Hebrew, Cyrillic or Devanagari
text. Every downstream statistic then collapsed to the "too short to judge"
branch, so the toolkit silently refused to analyse most of the world's writing.

Here the tokeniser is Unicode-aware and splits CJK runs into individual
characters (CJK has no spaces, and a character is the closest cheap analogue of
a word), while sentence splitting understands full-width terminators (。！？),
Arabic and Devanagari stops, and hard line breaks used as separators in lists.
"""
from __future__ import annotations

import re

# Ranges where each character is a reasonable "word" unit on its own.
_CJK = (
    r"぀-ヿ"      # kana
    r"㐀-䶿"      # CJK ext A
    r"一-鿿"      # CJK unified
    r"豈-﫿"      # compatibility ideographs
    r"가-힯"      # hangul syllables
)
_WORD_RE = re.compile(rf"[{_CJK}]|[^\W\d_][\w'’ـ-]*", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?။।۔؟…]+[\s]*|[。！？；]+|\n{1,}")


def words(text: str) -> list[str]:
    """Lower-cased word tokens; each CJK ideograph/syllable counts as one."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def sentences(text: str) -> list[str]:
    """Split on Latin, CJK, Arabic and Devanagari terminators plus newlines."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks; falls back to single-newline blocks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) > 1:
        return blocks
    return [b.strip() for b in text.split("\n") if b.strip()]


def is_cjk_dominant(text: str) -> bool:
    """True when most letters are CJK, where word-level stats behave differently."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cjk = sum(1 for c in letters if re.match(rf"[{_CJK}]", c))
    return cjk / len(letters) > 0.5
