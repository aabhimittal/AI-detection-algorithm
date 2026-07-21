"""Statistical watermark detector for LLM text (green-list / z-test).

WHY THIS IS DIFFERENT
---------------------
Every other text detector here is *post-hoc*: it looks at text it was never
meant to mark and guesses. Watermarking is *proactive* — the model provider
biases generation so the output secretly encodes a signal, and anyone with the
key can detect it with a clean statistical test and a false-positive rate you
can bound mathematically. This is the approach behind Kirchenbauer et al. (2023)
and, in spirit, Google's SynthID-Text.

THE GREEN-LIST SCHEME
---------------------
At each step, seed a PRNG with the *previous* token, then pseudo-randomly split
the vocabulary into a "green list" (a fraction gamma of tokens) and a "red
list". A watermarked model adds a small bias so it prefers green tokens.
Un-watermarked text hits green tokens only at the chance rate gamma.

Detection is a one-proportion z-test on the count of green tokens:

    z = (|s|_green - gamma * T) / sqrt(T * gamma * (1 - gamma))

where T is the number of scored tokens. A large positive z means far more green
tokens than chance => watermarked => machine-generated. z is converted to a
p-value and a [0, 1] suspicion score.

IMPLEMENTATION NOTE
-------------------
A faithful detector needs the *exact* tokenizer and vocab-hashing the generator
used. To stay dependency-free and self-contained, this operates at the WORD
level and hashes with blake2b — the algorithm is identical, only the tokenization
differs. Pass a `hash_key` matching the generator's key to detect real
watermarks; the default key is a placeholder for demonstration/testing.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Optional


def _is_green(prev_token: str, token: str, gamma: float, key: int) -> bool:
    """Deterministically decide whether `token` is on the green list.

    The list is a function of the previous token (context) + a secret key, so
    it's reproducible for the detector but unpredictable without the key —
    exactly the property the watermark relies on.
    """
    seed = f"{key}:{prev_token}".encode()
    # Hash (context) to a stream, then hash the candidate token against it and
    # map to [0, 1). token is green iff that value < gamma.
    h = hashlib.blake2b(seed + b"|" + token.encode(), digest_size=8).digest()
    frac = int.from_bytes(h, "big") / float(1 << 64)
    return frac < gamma


def watermark_score(
    text: str,
    gamma: float = 0.5,
    hash_key: int = 15485863,
    z_threshold: float = 4.0,
) -> Optional[float]:
    """Return P(watermarked) in [0, 1] from a green-list z-test.

    Args:
        text:        passage to test.
        gamma:       green-list fraction the generator used (default 0.5).
        hash_key:    secret key the generator watermarked with. MUST match to
                     detect a real watermark; the default is a placeholder.
        z_threshold: z at which we consider detection near-certain. Maps the
                     unbounded z to [0, 1] (z=0 -> 0.5*... , z>=thresh -> ~1).

    Returns:
        Score in [0, 1], or None if there aren't enough tokens to test.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if len(tokens) < 2:
        return None

    # Count green tokens over each (prev -> curr) transition.
    green = 0
    total = 0
    for prev, cur in zip(tokens[:-1], tokens[1:]):
        total += 1
        if _is_green(prev, cur, gamma, hash_key):
            green += 1

    # One-proportion z-statistic under H0 "no watermark" (green rate = gamma).
    expected = gamma * total
    std = math.sqrt(total * gamma * (1.0 - gamma)) + 1e-9
    z = (green - expected) / std

    # Convert z to a bounded, monotone suspicion score. Only positive z (more
    # green than chance) is evidence for a watermark, so negative z -> 0. The
    # score is the fraction of the way to `z_threshold`, the z at which
    # detection is treated as near-certain (z>=4 corresponds to a false-positive
    # rate ~3e-5 under the null, the standard operating point for green-list
    # watermarking). We deliberately do NOT use 1-p_value as the score: a p-value
    # is not P(watermarked), and a mild z~1 (p~0.16, ordinary null noise) must
    # not read as high suspicion.
    return float(max(0.0, min(1.0, z / z_threshold)))
