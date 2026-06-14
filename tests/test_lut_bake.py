"""Tests for the RF -> baked LUT fast path and trilinear apply."""

from __future__ import annotations

import time

import numpy as np

from photo_style.style_model import (
    DEFAULT_RF_LUT_SIZE,
    apply_cluster_transform,
    apply_rgb_lut,
    bake_rf_lut,
    fit_cluster_transform,
    sample_pair_pixels,
)


def _make_pair(rng: np.random.Generator, size=(128, 128)) -> tuple[np.ndarray, np.ndarray]:
    src = rng.integers(40, 200, size=(*size, 3), dtype=np.uint8)
    tgt = np.clip(src.astype(np.float32) * np.array([1.18, 1.0, 0.82]) + 8.0, 0, 255).astype(np.uint8)
    return src, tgt


def _rf_transform(rng: np.random.Generator):
    pair_pixels = []
    for _ in range(2):
        src, tgt = _make_pair(rng)
        s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=1500, rng=rng)
        pair_pixels.append((s_pix, t_pix))
    return fit_cluster_transform(
        pair_pixels, fit_rf=True, rf_n_estimators=12, rf_max_samples=3000, seed=0
    )


def test_fit_cluster_transform_bakes_lut() -> None:
    rng = np.random.default_rng(0)
    transform = _rf_transform(rng)
    assert transform.rf_lut is not None
    assert transform.rf_lut_size == DEFAULT_RF_LUT_SIZE
    assert transform.rf_lut.shape == (DEFAULT_RF_LUT_SIZE, DEFAULT_RF_LUT_SIZE, DEFAULT_RF_LUT_SIZE, 3)
    assert transform.rf_lut.min() >= 0.0 and transform.rf_lut.max() <= 1.0


def test_method_a_transform_has_no_lut() -> None:
    rng = np.random.default_rng(0)
    src, tgt = _make_pair(rng)
    s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=800, rng=rng)
    transform = fit_cluster_transform([(s_pix, t_pix)], fit_rf=False)
    assert transform.rf_lut is None
    assert transform.pixel_rf is None


def test_baked_lut_matches_direct_rf_closely() -> None:
    """Applying via the baked LUT should be close to running the RF per pixel."""
    rng = np.random.default_rng(0)
    transform = _rf_transform(rng)
    test_img = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)

    # Fast path (uses baked LUT via getattr).
    out_lut = apply_cluster_transform(test_img, transform, use_rf=True)

    # Force the per-pixel RF path by temporarily dropping the baked LUT.
    saved = transform.rf_lut
    transform.rf_lut = None
    try:
        out_rf = apply_cluster_transform(test_img, transform, use_rf=True)
    finally:
        transform.rf_lut = saved

    mean_abs_diff = float(np.mean(np.abs(out_lut.astype(np.int16) - out_rf.astype(np.int16))))
    # Trilinear interpolation of a 33^3 grid is a very close approximation.
    assert mean_abs_diff < 4.0


def test_apply_rgb_lut_identity_lut_is_noop() -> None:
    """An identity LUT should return the input essentially unchanged."""
    size = 33
    axis = np.linspace(0, 1, size, dtype=np.float32)
    R, G, B = np.meshgrid(axis, axis, axis, indexing="ij")
    identity = np.stack([R, G, B], axis=-1).astype(np.float32)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    out = apply_rgb_lut(img, identity)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert float(np.mean(np.abs(out.astype(np.int16) - img.astype(np.int16)))) < 1.5


def test_baked_lut_path_is_faster_than_per_pixel() -> None:
    """Sanity: on a reasonably large image the LUT path beats per-pixel RF."""
    rng = np.random.default_rng(0)
    transform = _rf_transform(rng)
    big = rng.integers(40, 200, size=(512, 512, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    apply_cluster_transform(big, transform, use_rf=True)  # LUT path
    lut_time = time.perf_counter() - t0

    saved = transform.rf_lut
    transform.rf_lut = None
    try:
        t0 = time.perf_counter()
        apply_cluster_transform(big, transform, use_rf=True)  # per-pixel path
        rf_time = time.perf_counter() - t0
    finally:
        transform.rf_lut = saved

    assert lut_time < rf_time


def test_bake_rf_lut_arbitrary_size() -> None:
    rng = np.random.default_rng(0)
    transform = _rf_transform(rng)
    lut17 = bake_rf_lut(transform.pixel_rf, 17)
    assert lut17.shape == (17, 17, 17, 3)
