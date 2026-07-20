# AI-detection-algorithm

A practical, **well-commented** toolkit for detecting AI-generated **images,
text, and video**. It implements the main *classes* of detection approach — from
model-free forensic heuristics to learned classifiers — and shows how to combine
several weak signals into one decision.

> **Detection is an arms race.** Generators improve constantly, so no single
> detector is reliable on its own. Every module here documents *what* it
> measures and *why* that betrays synthetic content, and returns a calibratable
> score in `[0, 1]`. The intended usage is to **ensemble** several signals — see
> `examples/demo.py`.

## Approaches implemented

### Images (`src/image_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `frequency_analysis` | GAN/diffusion upsamplers leave periodic peaks in the Fourier spectrum; real photos have a smooth 1/f spectrum. | numpy |
| `error_level_analysis` | Re-compressing a JPEG reveals regions with a different compression history (edits/splices/composites). | Pillow |
| `metadata_analysis` | Missing camera EXIF or a generative-tool `Software` tag. Cheap tie-breaker. | Pillow |
| `cnn_detector` | Learned classifier — strongest single signal; bring your own weights. | torch *(optional)* |

### Text (`src/text_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `perplexity` | LLM text is low-perplexity and low-"burstiness" under an LLM (the GPTZero intuition). | transformers *(optional)* |
| `stylometry` | Model-free style stats: lexical diversity, sentence-length variance, connective density. | none |
| `detectgpt` | Machine text sits at a local maximum of model log-prob; perturbations lower it (DetectGPT curvature). | transformers *(optional)* |

### Video (`src/video_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `temporal_consistency` | Deepfakes flicker: incoherent high-frequency change between frames. | opencv *(optional)* |
| `face_warping` | Face-swap blend seams + unnatural blink rate. | opencv *(optional)* |
| `frame_frequency` | Apply the image spectral detector across sampled frames. | opencv *(optional)* |

Heavy dependencies (torch, transformers, opencv) are **optional**: each detector
degrades gracefully and returns `None` if its dependency is missing, so the
classical detectors run with just `numpy`, `scipy`, and `Pillow`.

## Install

```bash
pip install -r requirements.txt   # torch/transformers/opencv lines are optional
```

## Usage

```bash
python examples/demo.py --image path/to/pic.jpg
python examples/demo.py --text  "the passage you want to test..."
python examples/demo.py --video path/to/clip.mp4
```

Or from Python:

```python
from src.image_detection import spectral_score, ela_score, combine_scores

scores = {"frequency": spectral_score("pic.jpg"), "ela": ela_score("pic.jpg")}
print(combine_scores(scores))   # -> P(AI-generated) in [0, 1]
```

## Tests

```bash
pip install pytest
pytest tests/          # runs with only numpy/scipy/Pillow installed
```

## Limitations (read these)

- Scores are **heuristic** and the built-in thresholds are hand-tuned, not
  calibrated on your data. Calibrate on a labeled set before trusting verdicts.
- Every signal here is defeatable in isolation (JPEG recompression fools
  frequency analysis; "humanizer" tools raise text perplexity; metadata is
  trivially forged). **Ensemble, and treat output as evidence, not proof.**
- `cnn_detector` ships **no weights** — it demonstrates the interface; supply a
  trained model for real accuracy.

## Layout

```
src/
  image_detection/   frequency, ELA, metadata, CNN
  text_detection/    perplexity, stylometry, detectgpt
  video_detection/   temporal, face-warping, frame-frequency
  utils/             shared preprocessing
examples/demo.py     ensemble CLI
tests/               dependency-free property tests
```
