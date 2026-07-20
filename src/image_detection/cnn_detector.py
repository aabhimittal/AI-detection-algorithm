"""Learned CNN detector for AI-generated images (strongest single signal).

WHY THIS WORKS
--------------
Classical detectors look for one hand-picked artifact. A convolutional network
trained on {real, generated} pairs learns *whatever* low-level statistics
distinguish the two — subtle color-channel correlations, texture regularities,
and upsampling fingerprints humans can't name. State-of-the-art detectors
(e.g. the "CNN-generated images are surprisingly easy to spot" line of work)
are exactly this: a ResNet-style classifier trained with aggressive JPEG/blur
augmentation so it generalizes across generators.

This module is a *thin, dependency-optional wrapper*. Training weights are not
shipped (they'd be large and quickly outdated), so:
  * If torch + a weights file are available, it runs a real forward pass.
  * Otherwise `cnn_score` raises a clear error telling you what to install /
    provide, and the rest of the toolkit keeps working without it.

Bring your own backbone/weights, or fine-tune one on your target generators.
"""
from __future__ import annotations

import numpy as np

try:  # Optional heavy dependency — the toolkit runs without it.
    import torch
    import torch.nn as nn
    from torchvision import transforms
    _TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _TORCH = False


# --- A small, honest reference backbone -------------------------------------
# Not a SOTA model — a compact CNN so the class is runnable end-to-end and shows
# the interface. Replace with a ResNet-50 (or your fine-tune) for real accuracy.
if _TORCH:

    class TinyDetector(nn.Module):
        """3-block conv net -> single logit (real vs generated)."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, 1)  # 1 logit; sigmoid -> P(generated)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.features(x).flatten(1)
            return self.head(z)


def cnn_score(path: str, weights: str | None = None, device: str = "cpu") -> float:
    """Return P(AI-generated) in [0, 1] from a learned CNN.

    Args:
        path:    image file to score.
        weights: path to a state_dict for TinyDetector (or your own model).
                 If None and no weights are found, we still run the *untrained*
                 network so the plumbing is demonstrable — but the number is
                 meaningless until you load real weights. A warning is printed.
        device:  "cpu" or "cuda".
    """
    if not _TORCH:
        raise RuntimeError(
            "cnn_score needs PyTorch + torchvision. Install them "
            "(`pip install torch torchvision`) or drop this signal from the "
            "ensemble — the classical detectors work without it."
        )

    model = TinyDetector().to(device).eval()
    if weights:
        # Load your trained parameters. map_location keeps it CPU-friendly.
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        print("[cnn_detector] WARNING: no weights supplied — output is random. "
              "Pass weights=<path> for a meaningful score.")

    # Standard ImageNet-style preprocessing so the input matches training.
    from PIL import Image
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():                      # inference only, no gradients
        logit = model(x).squeeze()
        prob = torch.sigmoid(logit).item()     # logit -> probability

    return float(np.clip(prob, 0.0, 1.0))
