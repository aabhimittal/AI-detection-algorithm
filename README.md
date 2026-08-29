# AI-detection-algorithm

A practical, **well-commented** toolkit for detecting AI-generated **images,
text, video, and audio**. It implements the main *classes* of detection
approach — from model-free forensic heuristics to learned classifiers — shows
how to combine several weak signals into one decision, and ships an evaluation
harness so you can calibrate the thresholds on your own data.

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
| `noise_residual` | Real sensors imprint white PRNU noise; generators are over-smooth or leave structured residuals. | scipy |
| `color_statistics` | Demosaicing ties camera color channels together; generators reproduce the co-occurrence stats imperfectly. | scipy |
| `cfa_periodicity` | A camera's Bayer colour-filter array leaves a 2x2 demosaicing lattice (a large Nyquist-frequency ridge in the high-pass residual); fully synthesised pixels have none. | numpy |
| `cnn_detector` | Learned classifier — strongest single signal; bring your own weights. | torch *(optional)* |

### Text (`src/text_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `perplexity` | LLM text is low-perplexity and low-"burstiness" under an LLM (the GPTZero intuition). | transformers *(optional)* |
| `stylometry` | Model-free style stats: lexical diversity, sentence-length variance, connective density. | none |
| `detectgpt` | Machine text sits at a local maximum of model log-prob; perturbations lower it (DetectGPT curvature). | transformers *(optional)* |
| `watermark` | **Proactive** green-list z-test (Kirchenbauer-style): detects the secret bias a watermarking model embeds, with a bounded false-positive rate. | none |
| `unicode_forensics` | Code-point forensics: invisible zero-width marks, homoglyph swaps from "humanizer" evasion tools, and the uniform typography of generated prose. Independent of wording, so it survives paraphrase attacks. | none |
| `structure` | Discourse shape: phrase recycling (distinct-3), repeated sentence openers, the three-item-list habit, answer scaffolding, assistant register phrases. Signals combine noisy-OR, so several weak tells compound. | none |

Text tokenisation is Unicode-aware: CJK, Arabic, Devanagari and Cyrillic
passages are analysed rather than silently skipped.

### Video (`src/video_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `temporal_consistency` | Deepfakes flicker: incoherent high-frequency change between frames. | opencv *(optional)* |
| `face_warping` | Face-swap blend seams + unnatural blink rate. | opencv *(optional)* |
| `frame_frequency` | Apply the image spectral detector across sampled frames. | opencv *(optional)* |

### Audio (`src/audio_detection/`)
| Module | Idea | Deps |
|---|---|---|
| `spectral_artifacts` | Neural vocoders leave a flat HF roll-off + periodic hop energy (the audio analogue of image upsampling). | numpy |
| `silence_stats` | Real recordings have a non-zero noise floor and irregular pauses; TTS gaps are often digitally clean and uniform. | numpy |

### Cross-media (`src/provenance.py`, `src/confidence.py`)
| Module | Idea | Deps |
|---|---|---|
| `provenance` | Reads C2PA / Content Credentials, XMP and PNG generator chunks. The only channel that can be decisive **in both directions**: `digitalSourceType=trainedAlgorithmicMedia` proves synthesis, `digitalCapture` is evidence of a real camera. Works bytewise on any file, so it covers video and audio too. | Pillow |
| `confidence` | Calibrated fusion: weighted combination, detector-agreement as an uncertainty proxy, hard-evidence override, ranked explanations, and **abstention** (`INCONCLUSIVE`) when the evidence is thin or contradictory. | none |

A score of exactly `0.5` means *"this detector found no evidence either way"*
and is excluded from the fusion rather than averaged in as a vote — otherwise a
silent detector would fake extra breadth of evidence.

Audio uses only stdlib `wave` + numpy (PCM WAV in; convert other formats with ffmpeg first).

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
python examples/demo.py --audio path/to/voice.wav
python examples/demo.py --auto  path/to/anything   # infers the media type
```

Or from Python — the **unified dispatcher** auto-routes by media type and runs
every applicable detector:

```python
from src.detector import detect

result = detect("pic.jpg")                       # media type inferred
result = detect("some passage...", media_type="text")
print(result["combined"], result["verdict"])     # e.g. 0.31 LIKELY AUTHENTIC
print(result["scores"])                          # per-detector breakdown
```

### Trained ensemble (better than the naive mean)

`combine_scores` averages detectors equally. Once you have labeled data, learn a
weighted fusion instead — a logistic regression that discovers which detectors
to trust and outputs a calibrated probability:

```python
from src.ensemble import LogisticEnsemble

ens = LogisticEnsemble(["frequency", "ela", "metadata", "noise_residual", "color_statistics"])
ens.fit(list_of_score_dicts, labels)     # labels: 1=AI, 0=authentic
ens.predict_one({"frequency": 0.7, "ela": 0.2})   # missing signals auto-imputed
ens.weights()                             # inspect learned per-detector weights
ens.save("model.json")                    # LogisticEnsemble.load(...) to restore
```

### Batch scoring a directory

Score a whole folder and export JSON/CSV for triage; pass labels to get an
end-to-end AUROC on your own data in one command:

```bash
python -m src.batch path/to/folder --csv results.csv --json results.json
python -m src.batch path/to/folder --labels labels.json   # -> AUROC + best threshold
```

### Calibrating thresholds on your data

The built-in 0.5 / 0.75 thresholds are hand-set. Replace them with values fit to
a labeled set using `src/evaluation`:

```python
import numpy as np
from src.detector import detect
from src.evaluation import roc_auc, best_threshold

labels = np.array([...])   # 1 = AI-generated, 0 = authentic
scores = np.array([detect(p)["combined"] for p in paths])
print("AUROC:", roc_auc(labels, scores))         # threshold-free separability
print("best cut:", best_threshold(labels, scores))
```

## Reading a result

```python
from src.detector import detect, detect_many

r = detect("upload.bin")          # media type inferred from magic bytes
r["combined"]            # 0.0-1.0, weighted fusion
r["verdict"]             # LIKELY AI-GENERATED / POSSIBLY ... / LIKELY AUTHENTIC
r["calibrated_verdict"]  # the same, or INCONCLUSIVE when we should not commit
r["confidence"]          # 0.0-1.0 from agreement, breadth and decisiveness
r["explanation"]         # signals ranked by how much they moved the verdict
r["provenance"]          # {"generator": "ComfyUI", "declared": True, ...}
r["errors"]              # per-detector failures; one bad detector never aborts

detect_many([...])       # one row per input, failures recorded not raised
```

Bad input fails loudly and specifically: `FileNotFoundError` (with a hint to
pass `media_type="text"`), `IsADirectoryError` (pointing at the batch API),
`ValueError` for empty files, empty text and unknown media types, `TypeError`
for non-string input.

## Tests

```bash
pip install pytest
pytest tests/          # runs with only numpy/scipy/Pillow installed
```

`tests/test_edge_cases.py` covers the degenerate inputs a deployment meets on
day one: empty and whitespace text, punctuation/emoji/digit-only passages, CJK,
Arabic, Devanagari and Cyrillic, RTL overrides and control characters, missing
files, directories, zero-byte and truncated files, extensionless uploads and
lying extensions, 1x1 and 1xN images, greyscale/RGBA/palette/CMYK modes, and
Unicode filenames.

## Limitations (read these)

- Scores are **heuristic** and the built-in thresholds are hand-tuned, not
  calibrated on your data. Calibrate on a labeled set before trusting verdicts.
- Every signal here is defeatable in isolation (JPEG recompression fools
  frequency analysis; "humanizer" tools raise text perplexity; metadata is
  trivially forged). **Ensemble, and treat output as evidence, not proof.**
- Absence of provenance metadata proves **nothing** — it is stripped by every
  social platform. Only its *presence* is treated as high-confidence evidence.
- `cfa_periodicity` also fires on legitimately re-encoded, rescaled or
  screenshotted photos, which destroy the demosaicing lattice. It is
  corroboration, never a verdict on its own.
- `cnn_detector` ships **no weights** — it demonstrates the interface; supply a
  trained model for real accuracy.

## Layout

```
src/
  detector.py        unified detect()/detect_many() dispatcher (extension +
                     magic-byte routing, per-detector fault isolation)
  confidence.py      calibrated fusion, uncertainty, abstention, explanations
  provenance.py      C2PA / Content Credentials / generator-metadata scanning
  ensemble.py        LogisticEnsemble — trained weighted fusion of detectors
  batch.py           directory batch scorer + CLI (JSON/CSV, optional AUROC)
  image_detection/   frequency, ELA, metadata, noise-residual, color-stats,
                     CFA/demosaicing periodicity, CNN
  text_detection/    perplexity, stylometry, detectgpt, watermark,
                     unicode forensics, discourse structure
  video_detection/   temporal, face-warping, frame-frequency
  audio_detection/   vocoder spectral artifacts, silence/noise-floor stats
  evaluation/        AUROC, best-threshold, confusion/precision/recall
  utils/             shared preprocessing
examples/demo.py     ensemble CLI (--image/--text/--video/--audio/--auto/--stdin)
tests/               dependency-free property tests
```
