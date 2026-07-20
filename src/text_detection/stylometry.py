"""Model-free stylometric detector.

WHY THIS WORKS
--------------
Before neural language models, authorship attribution relied on *stylometry*:
statistical fingerprints of writing style that authors can't easily control.
Several of these features differ systematically between human and current LLM
prose:

  * Type-Token Ratio (lexical diversity): LLMs reuse a slightly narrower,
    "safe" vocabulary within a passage.
  * Sentence-length variance: humans mix very short and very long sentences;
    LLM output clusters around a comfortable medium length (low variance) —
    the same "low burstiness" idea, measured without a model.
  * Function-word rate: the density of connectives ("however", "moreover",
    "additionally") tends to run higher in LLM text.
  * Punctuation regularity.

None of these is decisive; together they're a cheap, offline sanity check that
needs no GPU and no model download. Returns P(AI-generated) in [0, 1].
"""
from __future__ import annotations

import re
import statistics

# A small set of connectives that LLMs over-use relative to casual human prose.
_LLM_FAVORED = {
    "however", "moreover", "furthermore", "additionally", "consequently",
    "therefore", "thus", "overall", "importantly", "notably", "essentially",
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def stylometry_score(text: str) -> float:
    """Combine three model-free stylometric signals into P(AI) in [0, 1]."""
    words = _words(text)
    sents = _sentences(text)
    if len(words) < 20 or len(sents) < 2:
        # Too little text to be statistically meaningful -> neutral.
        return 0.5

    # 1. Lexical diversity (type-token ratio). LOW diversity -> more machine-y.
    ttr = len(set(words)) / len(words)
    #    ttr ~0.4 (repetitive) .. 0.7 (varied); invert so low ttr => high score.
    div_signal = max(0.0, min(1.0, (0.7 - ttr) / 0.3))

    # 2. Sentence-length uniformity. LOW variance -> more machine-y.
    lengths = [len(_words(s)) for s in sents]
    cv = statistics.pstdev(lengths) / (statistics.mean(lengths) + 1e-6)  # coeff of variation
    #    cv ~0.2 (uniform, machine) .. 0.8 (bursty, human); invert.
    uniform_signal = max(0.0, min(1.0, (0.6 - cv) / 0.6))

    # 3. Connective density. HIGH -> more machine-y.
    conn = sum(1 for w in words if w in _LLM_FAVORED) / len(words)
    #    ~0.5% is ordinary; >2% is notably heavy on connectives.
    conn_signal = max(0.0, min(1.0, conn / 0.02))

    # Equal-weight average of the three.
    return float((div_signal + uniform_signal + conn_signal) / 3.0)
