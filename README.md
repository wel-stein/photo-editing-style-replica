# photo_style_ai

Learn a personal photo editing style from before/after pairs and apply it to new photos.
Exports a 33x33x33 `.cube` LUT usable in Lightroom, Photoshop, and DaVinci Resolve.

Runs entirely locally on CPU. No GPU, no cloud, no external APIs.

## Status

Phase 1 (scaffold + Reinhard smoke test). See `scripts/smoke_test.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Smoke test

```bash
python scripts/smoke_test.py path/to/original.jpg path/to/edited.jpg --out out.jpg
```

This loads a single before/after pair, fits Reinhard color transfer statistics
(mean/std in LAB), applies the transfer to the original, and writes the result.
