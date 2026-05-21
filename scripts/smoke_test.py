"""Phase 1 smoke test: load one before/after pair, fit Reinhard stats, apply, save.

Usage:
    python scripts/smoke_test.py <original> <edited> --out out.jpg

If no arguments are provided, a synthetic pair is generated in memory so the
script can be exercised without any input files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from photo_style.io_utils import load_image_rgb, save_image_rgb
from photo_style.style_model import apply_reinhard, fit_reinhard


def _synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic synthetic before/after pair for self-test."""
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, size=(128, 128, 3), dtype=np.uint8)
    # "Edited" version: warmer + slightly more contrasty.
    edited = np.clip(base.astype(np.int16) * np.array([1.15, 1.00, 0.85]) + 5, 0, 255).astype(
        np.uint8
    )
    return base, edited


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 Reinhard smoke test")
    parser.add_argument("original", nargs="?", help="Path to original (unedited) image")
    parser.add_argument("edited", nargs="?", help="Path to edited image (style target)")
    parser.add_argument("--out", default="out.jpg", help="Output path for the styled image")
    args = parser.parse_args()

    if args.original and args.edited:
        src = load_image_rgb(args.original)
        tgt = load_image_rgb(args.edited)
        logger.info("Loaded pair: src={} tgt={}", src.shape, tgt.shape)
    else:
        logger.warning("No paths given - using a synthetic pair so the test still runs.")
        src, tgt = _synthetic_pair()

    stats = fit_reinhard(src, tgt)
    logger.info(
        "Reinhard stats: src_mean={} tgt_mean={}",
        np.round(stats.src_mean, 2).tolist(),
        np.round(stats.tgt_mean, 2).tolist(),
    )

    styled = apply_reinhard(src, stats)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_rgb(styled, out_path)
    size_kb = out_path.stat().st_size / 1024
    logger.success(
        "Wrote {} (shape={}, dtype={}, range=[{},{}], file={:.1f} KB)",
        out_path,
        styled.shape,
        styled.dtype,
        int(styled.min()),
        int(styled.max()),
        size_kb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
