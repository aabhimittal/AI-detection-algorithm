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

import statistics

from ._tokenize import is_cjk_dominant, sentences as _sentences, words as _words

# A small set of connectives that LLMs over-use relative to casual human prose.
_LLM_FAVORED = {
    "however", "moreover", "furthermore", "additionally", "consequently",
    "therefore", "thus", "overall", "importantly", "notably", "essentially",
}


def stylometry_score(text: str) -> float:
    """Combine three model-free stylometric signals into P(AI) in [0, 1]."""
    words = _words(text)
    sents = _sentences(text)
    if len(words) < 20 or len(sents) < 2:
        # Too little text to be statistically meaningful -> neutral.
        return 0.5

    # CJK is written without spaces, so a "word" here is a single character and
    # the Latin-tuned type-token and connective thresholds do not transfer.
    # Fall back to the two script-neutral signals in that case.
    cjk = is_cjk_dominant(text)

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

    if cjk:
        # Character-level TTR runs much lower; drop the Latin-calibrated
        # diversity and connective terms rather than reporting a biased score.
        return float(uniform_signal)

    # Equal-weight average of the three.
    return float((div_signal + uniform_signal + conn_signal) / 3.0)
