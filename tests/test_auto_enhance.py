"""Tests for the Quick Auto-Enhance pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from photo_style.auto_enhance import (
    EnhanceParams,
    apply_local_contrast,
    auto_enhance,
    auto_exposure,
    auto_white_balance,
    boost_saturation,
    lift_shadows,
    recover_highlights,
    suggest_defaults,
)


def _mean_L(rgb_u8: np.ndarray) -> float:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[..., 0].mean())


def _solid(color=(128, 128, 128), size=(64, 64)) -> np.ndarray:
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def _random_rgb(rng: np.random.Generator, lo=40, hi=200, size=(96, 96)) -> np.ndarray:
    return rng.integers(lo, hi, size=(*size, 3), dtype=np.uint8)


def test_auto_white_balance_identity_at_zero() -> None:
    rgb = _random_rgb(np.random.default_rng(0))
    out = auto_white_balance(rgb, strength=0.0)
    assert np.array_equal(out, rgb)


def test_auto_white_balance_neutralizes_color_cast() -> None:
    """Strong red-tinted input should land closer to gray after WB."""
    rng = np.random.default_rng(0)
    base = rng.integers(80, 160, size=(96, 96, 3), dtype=np.uint8)
    tinted = np.clip(base.astype(np.float32) * np.array([1.4, 1.0, 0.7]), 0, 255).astype(np.uint8)
    out = auto_white_balance(tinted, strength=1.0)
    tinted_means = tinted.reshape(-1, 3).mean(axis=0)
    out_means = out.reshape(-1, 3).mean(axis=0)
    # After gray world, channel means should be tighter together.
    assert (out_means.max() - out_means.min()) < (tinted_means.max() - tinted_means.min())


def test_auto_exposure_brightens_dark_image() -> None:
    dark = _solid(color=(40, 40, 40))
    out = auto_exposure(dark, strength=1.0)
    assert _mean_L(out) > _mean_L(dark)


def test_auto_exposure_darkens_bright_image() -> None:
    bright = _solid(color=(220, 220, 220))
    out = auto_exposure(bright, strength=1.0)
    assert _mean_L(out) < _mean_L(bright)


def test_auto_exposure_no_change_at_zero_strength() -> None:
    img = _solid(color=(60, 60, 60))
    out = auto_exposure(img, strength=0.0)
    assert np.array_equal(out, img)


def test_lift_shadows_brightens_dark_regions_only() -> None:
    rng = np.random.default_rng(0)
    # Half dark, half bright
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:, :64] = rng.integers(10, 40, size=(64, 64, 3), dtype=np.uint8)  # shadows
    img[:, 64:] = rng.integers(200, 250, size=(64, 64, 3), dtype=np.uint8)  # highlights

    lifted = lift_shadows(img, strength=1.0)
    # Shadow half should brighten meaningfully.
    assert lifted[:, :64].mean() > img[:, :64].mean() + 5
    # Highlight half should stay close.
    assert abs(lifted[:, 64:].mean() - img[:, 64:].mean()) < 5


def test_recover_highlights_darkens_bright_regions_only() -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:, :64] = rng.integers(10, 40, size=(64, 64, 3), dtype=np.uint8)
    img[:, 64:] = rng.integers(200, 250, size=(64, 64, 3), dtype=np.uint8)

    recovered = recover_highlights(img, strength=1.0)
    # Highlight half should darken.
    assert recovered[:, 64:].mean() < img[:, 64:].mean() - 3
    # Shadow half should stay close.
    assert abs(recovered[:, :64].mean() - img[:, :64].mean()) < 5


def test_boost_saturation_increases_chroma() -> None:
    rgb = _solid(color=(180, 100, 100))  # reddish
    out = boost_saturation(rgb, strength=1.0)
    # Convert to HSV and compare saturation channel.
    s_before = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)[..., 1].mean()
    s_after = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)[..., 1].mean()
    assert s_after > s_before


def test_apply_local_contrast_increases_L_std_on_flat_image() -> None:
    rng = np.random.default_rng(0)
    flat = rng.integers(110, 140, size=(128, 128, 3), dtype=np.uint8)
    out = apply_local_contrast(flat, strength=1.0)
    bgr_a = cv2.cvtColor(flat, cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    L_before = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB)[..., 0].std()
    L_after = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2LAB)[..., 0].std()
    assert L_after > L_before


def test_auto_enhance_preserves_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    rgb = _random_rgb(rng, size=(96, 96))
    params = EnhanceParams(white_balance=0.4, exposure=0.3, shadows=0.4, highlights=0.3, local_contrast=0.2, saturation=0.3)
    out = auto_enhance(rgb, params)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_auto_enhance_with_all_zero_strengths_returns_unchanged() -> None:
    rng = np.random.default_rng(0)
    rgb = _random_rgb(rng)
    out = auto_enhance(rgb, EnhanceParams())  # all zeros
    assert np.array_equal(out, rgb)


def test_auto_enhance_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError):
        auto_enhance(np.zeros((64, 64), dtype=np.uint8), EnhanceParams())
    with pytest.raises(ValueError):
        auto_enhance(np.zeros((64, 64, 3), dtype=np.float32), EnhanceParams())


def test_suggest_defaults_underexposed_recommends_exposure() -> None:
    dark = _solid(color=(45, 45, 45))
    params = suggest_defaults(dark)
    assert params.exposure > 0


def test_suggest_defaults_overexposed_recommends_exposure() -> None:
    bright = _solid(color=(220, 220, 220))
    params = suggest_defaults(bright)
    assert params.exposure > 0


def test_suggest_defaults_neutral_skips_exposure() -> None:
    neutral = _solid(color=(128, 128, 128))
    params = suggest_defaults(neutral)
    assert params.exposure == 0.0


def test_suggest_defaults_flat_image_boosts_local_contrast() -> None:
    rng = np.random.default_rng(0)
    flat = rng.integers(120, 135, size=(128, 128, 3), dtype=np.uint8)
    params = suggest_defaults(flat)
    assert params.local_contrast > 0.5


def test_suggest_defaults_returns_params_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    rgb = _random_rgb(rng)
    params = suggest_defaults(rgb)
    for v in (params.white_balance, params.exposure, params.shadows,
              params.highlights, params.local_contrast, params.saturation):
        assert 0.0 <= v <= 1.0
