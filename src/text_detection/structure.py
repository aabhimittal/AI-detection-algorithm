"""Discourse-structure detector: how a passage is *organised*, not worded.

WHY THIS WORKS
--------------
Stylometry looks at word statistics and perplexity looks at token likelihood.
Both are degraded by paraphrasing. The shape of the argument survives it:

  * Phrase reuse. Assistant prose recycles n-grams within a passage far more
    than human prose ("distinct-3" ratio, the standard diversity metric from
    dialogue generation research).
  * Opener repetition. Consecutive sentences that begin with the same word or
    the same part-of-speech scaffold ("This ...", "It ...", "Additionally ...")
    reveal a template.
  * Tricolon habit. LLMs disproportionately produce three-item lists
    ("clear, concise, and correct"). Humans use two- and four-item lists too.
  * Scaffolding. Bullet lists, bold headings and a closing "In conclusion" in
    a short passage indicate a generated answer rather than natural writing.
  * Register tells. A small set of phrases ("it is important to note", "delve
    into", "a testament to", "plays a crucial role") is heavily over-represented
    in assistant output.
  * Paragraph uniformity. Generated paragraphs come out near-equal in length.

Returns P(AI-generated) in [0, 1]; short input returns the neutral 0.5.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from ._tokenize import paragraphs, sentences, words

_REGISTER_PHRASES = (
    "it is important to note", "it's important to note", "in conclusion",
    "delve into", "a testament to", "plays a crucial role", "plays a vital role",
    "when it comes to", "in today's", "navigating the", "the world of",
    "it is worth noting", "on the other hand", "in summary", "let's explore",
    "rich tapestry", "ever-evolving", "significantly enhance", "a wide range of",
)
_TRICOLON_RE = re.compile(r"\b[\w'’-]+,\s+[\w'’-]+,\s+(?:and|or)\s+[\w'’-]+\b", re.I)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+\S|\*\*[^*\n]{3,60}\*\*:?\s*$)", re.M)


def _distinct_ngram_ratio(toks: list[str], n: int) -> float:
    """Unique / total n-grams. 1.0 = no repetition, lower = more recycling."""
    if len(toks) <= n:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def structure_report(text: str) -> dict:
    """Raw evidence behind :func:`structure_score` (useful for explanations)."""
    toks = words(text)
    sents = sentences(text)
    paras = paragraphs(text)
    lower = text.lower()

    openers = [s.split()[0].lower() for s in sents if s.split()]
    opener_repeat = 0.0
    if len(openers) >= 3:
        top = Counter(openers).most_common(1)[0][1]
        opener_repeat = (top - 1) / (len(openers) - 1)

    para_lengths = [len(words(p)) for p in paras]
    para_uniformity = 0.0
    if len(para_lengths) >= 3 and statistics.mean(para_lengths) > 0:
        cv = statistics.pstdev(para_lengths) / statistics.mean(para_lengths)
        para_uniformity = max(0.0, 1.0 - cv / 0.5)

    return {
        "tokens": len(toks),
        "sentences": len(sents),
        "distinct_3": _distinct_ngram_ratio(toks, 3),
        "opener_repeat": opener_repeat,
        "tricolons": len(_TRICOLON_RE.findall(text)),
        "bullets": len(_BULLET_RE.findall(text)),
        "headings": len(_HEADING_RE.findall(text)),
        "register_hits": sum(lower.count(p) for p in _REGISTER_PHRASES),
        "paragraph_uniformity": para_uniformity,
    }


def structure_score(text: str) -> float:
    """Fuse the discourse-level signals into P(AI-generated) in [0, 1]."""
    if not isinstance(text, str):
        raise TypeError("structure_score expects str")

    r = structure_report(text)
    if r["tokens"] < 30 or r["sentences"] < 3:
        return 0.5                                   # not enough structure to read

    # 1. Phrase recycling. >=0.99 distinct trigrams is normal; 0.90 is repetitive.
    repetition = max(0.0, min(1.0, (0.99 - r["distinct_3"]) / 0.09))

    # 2. Repeated sentence openers (a third of sentences sharing one saturates).
    opener = max(0.0, min(1.0, r["opener_repeat"] / 0.34))

    # 3. Tricolons: both the rate (~1.5 per 100 tokens saturates) and the raw
    #    count matter. One three-item list is ordinary human writing; it is the
    #    *habit* that is diagnostic, so a single occurrence can only score half.
    tricolon = (min(1.0, (r["tricolons"] / max(r["tokens"], 1) * 100) / 1.5)
                * min(1.0, r["tricolons"] / 2.0))

    # 4. Answer scaffolding: bullets and headings in a short passage.
    scaffold = min(1.0, (r["bullets"] + 2 * r["headings"]) / 6.0)

    # 5. Register phrases (~3 per 100 tokens saturates; formal human writing
    #    uses one or two of these too, so a single hit must not saturate).
    register = (min(1.0, (r["register_hits"] / max(r["tokens"], 1) * 100) / 3.0)
                * min(1.0, r["register_hits"] / 2.0))

    # Noisy-OR rather than a weighted mean: these habits are *independent*
    # tells, so three of them firing at once should compound instead of being
    # averaged down by the ones a short passage cannot exhibit (paragraph
    # uniformity needs three paragraphs; repetition needs length). Each weight
    # is the signal's reliability on its own.
    signals = [repetition, opener, tricolon, register, r["paragraph_uniformity"]]
    reliability = [0.55, 0.30, 0.45, 0.50, 0.30]
    miss = 1.0
    for sig, rel in zip(signals, reliability):
        miss *= 1.0 - rel * sig
    base = 1.0 - miss
    # Scaffolding is corroborating, not decisive: it can lift the score by half
    # the remaining headroom at most.
    return float(min(1.0, base + (1.0 - base) * 0.35 * scaffold))
