"""Text AI-generation detectors.

Three complementary approaches:

    perplexity  -- LLM-written text is, by construction, high-probability under
                   an LLM, so it has LOW perplexity and LOW "burstiness"
                   (sentence-to-sentence surprise variance). Human text is
                   bumpier. This is the intuition behind GPTZero-style tools.
    stylometry  -- classic authorship features (type-token ratio, function-word
                   rates, sentence-length variance). Fast, model-free, works
                   even when you can't run an LLM.
    detectgpt   -- the strongest zero-shot idea: machine text sits at a local
                   maximum of the model's log-probability, so small paraphrases
                   almost always LOWER its probability. Human text does not have
                   this curvature. Needs a model to score + perturb.

As with images: fuse them. Each returns P(AI-generated) in [0, 1].
"""
from .perplexity import perplexity_score       # noqa: F401
from .stylometry import stylometry_score        # noqa: F401
from .watermark import watermark_score          # noqa: F401


def combine_text_scores(scores: dict[str, float]) -> float:
    """Equal-weight mean of available text signals, each in [0, 1]."""
    vals = [v for v in scores.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
