"""Tests for feature extraction against synthetic images."""

from __future__ import annotations

import numpy as np
import pytest

from photo_style.features import (
    NUM_HIST_BINS,
    PhotoFeatures,
    extract_features,
    feature_vector_dim,
)


def _solid(color=(128, 128, 128), size=(64, 64)) -> np.ndarray:
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def test_extract_returns_expected_vector_size() -> None:
    feats = extract_features(_solid())
    v = feats.to_vector()
    assert v.shape == (feature_vector_dim(),)
    assert v.dtype == np.float32
    assert feature_vector_dim() == 6 + 3 * NUM_HIST_BINS


def test_feature_names_match_vector_length() -> None:
    assert len(PhotoFeatures.feature_names()) == feature_vector_dim()


def test_histograms_normalized_and_nonnegative() -> None:
    feats = extract_features(_solid(color=(100, 150, 200)))
    for h in (feats.L_hist, feats.a_hist, feats.b_hist):
        assert h.shape == (NUM_HIST_BINS,)
        assert (h >= 0).all()
        assert abs(h.sum() - 1.0) < 1e-5


def test_brighter_image_has_higher_L_mean() -> None:
    dark = extract_features(_solid(color=(40, 40, 40)))
    bright = extract_features(_solid(color=(220, 220, 220)))
    assert bright.L_mean > dark.L_mean


def test_warm_image_has_higher_b_mean() -> None:
    """A warm (yellow) image should land higher on the LAB b-axis than neutral."""
    neutral = extract_features(_solid(color=(128, 128, 128)))
    warm = extract_features(_solid(color=(220, 180, 80)))
    assert warm.b_mean > neutral.b_mean


def test_red_image_has_higher_a_mean() -> None:
    neutral = extract_features(_solid(color=(128, 128, 128)))
    red = extract_features(_solid(color=(220, 80, 80)))
    assert red.a_mean > neutral.a_mean


def test_high_contrast_image_has_higher_L_std() -> None:
    flat = extract_features(_solid(color=(128, 128, 128)))
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    high_contrast = extract_features(checker)
    assert high_contrast.L_std > flat.L_std


def test_downsampling_preserves_global_stats() -> None:
    """Constant-color image: features should be ~identical before vs after downsampling."""
    color = (120, 80, 60)
    small = extract_features(_solid(color=color, size=(64, 64)))
    big = extract_features(_solid(color=color, size=(2048, 2048)))  # triggers downsample
    assert abs(small.L_mean - big.L_mean) < 1.0
    assert abs(small.a_mean - big.a_mean) < 1.0
    assert abs(small.b_mean - big.b_mean) < 1.0


def test_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError):
        extract_features(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_features(np.zeros((64, 64, 4), dtype=np.uint8))


def test_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError):
        extract_features(np.zeros((64, 64, 3), dtype=np.float32))
