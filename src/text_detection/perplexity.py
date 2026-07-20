"""Perplexity + burstiness detector (the GPTZero intuition).

WHY THIS WORKS
--------------
A language model assigns a probability to every next token. "Perplexity" is,
loosely, how *surprised* the model is by a text — the exponential of the average
negative log-likelihood per token. Because an LLM generates high-probability
continuations of itself, LLM-written text has LOW perplexity under an LLM.
Human writing wanders, makes unusual word choices, and is therefore more
surprising => HIGHER perplexity.

"Burstiness" captures the *variance* of that surprise across sentences. Humans
alternate short punchy sentences with long complex ones, so their per-sentence
perplexity swings a lot (high burstiness). LLM text is smoother and more
uniform (low burstiness).

So a text that is BOTH low-perplexity AND low-burstiness is likely machine-made.

CAVEATS (state them honestly):
  * Non-native human writing can also be low-perplexity.
  * Paraphrasing/"humanizer" tools raise perplexity on purpose to evade this.
  * The scoring model must be *comparable* to the suspected generator; scoring
    GPT-4 text with a tiny GPT-2 is weaker. This module defaults to GPT-2 for
    zero-setup use — upgrade the model id for better results.
"""
from __future__ import annotations

import math
from typing import Optional

try:  # Model-based -> optional heavy deps.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

# Lazily-loaded singletons so repeated calls don't reload the model.
_TOK = None
_MODEL = None


def _load(model_id: str):
    global _TOK, _MODEL
    if _TOK is None or _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_id).eval()
    return _TOK, _MODEL


def _text_perplexity(text: str, tok, model) -> float:
    """Perplexity of one string = exp(mean token-level cross-entropy).

    We feed the text as both input and label; the model's built-in loss IS the
    mean negative log-likelihood over tokens, so exp(loss) is the perplexity.
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        # labels=input_ids => the model computes next-token cross-entropy.
        loss = model(**ids, labels=ids["input_ids"]).loss
    return math.exp(loss.item())


def perplexity_score(text: str, model_id: str = "gpt2") -> Optional[float]:
    """Return P(AI-generated) in [0, 1] from low perplexity + low burstiness.

    Returns None (rather than a fabricated number) if transformers/torch are
    unavailable, so the caller can simply drop this signal from the ensemble.
    """
    if not _OK:
        return None

    tok, model = _load(model_id)

    # Split into sentences (naive but dependency-free) to measure burstiness.
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) < 2:
        sentences = [text]

    # Per-sentence perplexity.
    ppls = [_text_perplexity(s, tok, model) for s in sentences]
    mean_ppl = sum(ppls) / len(ppls)
    # Burstiness = standard deviation of sentence perplexities.
    var = sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)
    burstiness = math.sqrt(var)

    # Convert to a suspicion score. Low perplexity AND low burstiness -> high
    # score. We map each through a decreasing function and average.
    #   perplexity ~ [10 (machine) .. 100+ (human)]  -> 40 as the midpoint
    #   burstiness ~ [<10 (machine) .. 50+ (human)]   -> 25 as the midpoint
    ppl_signal = 1.0 / (1.0 + math.exp((mean_ppl - 40.0) / 15.0))   # sigmoid, low ppl -> ~1
    burst_signal = 1.0 / (1.0 + math.exp((burstiness - 25.0) / 10.0))
    return float((ppl_signal + burst_signal) / 2.0)
