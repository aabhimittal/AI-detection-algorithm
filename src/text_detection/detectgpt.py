"""DetectGPT-style curvature detector (strongest zero-shot text signal).

WHY THIS WORKS
--------------
Key insight from the DetectGPT paper (Mitchell et al., 2023): text sampled from
a model tends to occupy a *local maximum* of that model's log-probability
surface. If you take machine text and make many small paraphrase-style
perturbations, almost all of them move the text to a region the model finds LESS
probable — the log-prob curvature is negative.

Human text is not sitting on such a peak, so perturbations move its probability
up about as often as down — near-zero curvature.

The detector estimates that curvature:

    score = logP(original) - mean_i logP(perturbation_i)

A clearly positive gap (original much more probable than its neighbours) =>
machine-generated.

COST & DEPENDENCIES
-------------------
This needs (a) a scoring LM for log-probabilities and (b) a way to perturb text
— the paper uses a T5 mask-filling model. Both are optional heavy deps, so the
function returns None when they're missing. The perturbation here is a light,
dependency-free word-dropout stand-in with a hook to plug in real T5 masking;
swap it for mask-filling for paper-faithful accuracy.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

_TOK = None
_MODEL = None


def _load(model_id: str):
    global _TOK, _MODEL
    if _TOK is None or _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_id).eval()
    return _TOK, _MODEL


def _logprob(text: str, tok, model) -> float:
    """Total log-probability the model assigns to `text` (higher = more likely).

    The model's cross-entropy loss is the *mean negative* log-likelihood per
    token; multiply by token count and negate to recover total log-prob.
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    n = ids["input_ids"].shape[1]
    with torch.no_grad():
        loss = model(**ids, labels=ids["input_ids"]).loss
    return -loss.item() * n


def _default_perturb(text: str, rng_seed: int) -> str:
    """Cheap, deterministic stand-in perturbation: drop ~15% of words.

    Deterministic (seeded by index) so results are reproducible. This is a
    weaker perturbation than T5 mask-filling; pass your own `perturb_fn` for
    paper-grade behaviour.
    """
    words = text.split()
    if len(words) < 5:
        return text
    # Simple linear-congruential shuffle -> pick indices to drop without random.
    keep = [w for i, w in enumerate(words) if (i * 2654435761 + rng_seed) % 7 != 0]
    return " ".join(keep) if keep else text


def detectgpt_score(
    text: str,
    model_id: str = "gpt2",
    n_perturb: int = 10,
    perturb_fn: Optional[Callable[[str, int], str]] = None,
) -> Optional[float]:
    """Return P(AI-generated) in [0, 1] from log-probability curvature.

    Args:
        text:       passage to test.
        model_id:   causal LM used to score log-probabilities.
        n_perturb:  number of perturbed variants to average over. More = more
                    stable estimate, linearly more compute.
        perturb_fn: (text, seed) -> perturbed text. Defaults to word-dropout;
                    supply T5 mask-filling for best accuracy.
    """
    if not _OK:
        return None

    tok, model = _load(model_id)
    perturb_fn = perturb_fn or _default_perturb

    orig = _logprob(text, tok, model)
    perturbed = [_logprob(perturb_fn(text, i + 1), tok, model) for i in range(n_perturb)]
    mean_p = sum(perturbed) / len(perturbed)

    # Normalize the gap by the spread of perturbed log-probs (a z-score-like
    # discrepancy), matching the paper's variance normalization. Large positive
    # discrepancy => machine.
    var = sum((p - mean_p) ** 2 for p in perturbed) / len(perturbed)
    std = math.sqrt(var) + 1e-6
    discrepancy = (orig - mean_p) / std

    # Map the open-ended discrepancy to [0, 1] with a sigmoid. Discrepancy ~0
    # (human) -> 0.5; strongly positive (machine) -> ~1.
    return float(1.0 / (1.0 + math.exp(-discrepancy)))
