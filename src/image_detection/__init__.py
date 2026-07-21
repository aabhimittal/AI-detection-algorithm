"""Image AI-generation / manipulation detectors.

Four complementary approaches, weakest-assumption to strongest-model:

    frequency_analysis  -- GAN/diffusion upsamplers leave periodic spectral
                           peaks; real camera images have a smooth 1/f spectrum.
    error_level_analysis-- re-compressing a JPEG reveals regions that were
                           edited/spliced (classic forensics, also flags some
                           synthetic composites).
    metadata_analysis   -- missing camera EXIF / tell-tale software tags.
    cnn_detector        -- learned classifier; strongest but needs weights.

Ensemble them — see `combine_scores`.
"""
from .frequency_analysis import spectral_score      # noqa: F401
from .error_level_analysis import ela_score          # noqa: F401
from .metadata_analysis import metadata_score        # noqa: F401
from .noise_residual import noise_residual_score     # noqa: F401
from .color_statistics import color_statistics_score  # noqa: F401


def combine_scores(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted average of individual detector scores, each in [0, 1].

    A higher combined score => more likely AI-generated/manipulated. Defaults
    weight every provided signal equally. Keeping the fusion this simple is
    deliberate: it's transparent and hard to overfit. Swap in a trained
    logistic-regression meta-classifier for production.
    """
    if not scores:
        return 0.0
    weights = weights or {k: 1.0 for k in scores}
    total = sum(weights.get(k, 0.0) for k in scores)
    if total == 0:
        return 0.0
    return sum(scores[k] * weights.get(k, 0.0) for k in scores) / total
