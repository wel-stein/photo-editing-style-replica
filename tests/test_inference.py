"""Tests for the apply_profile inference entry point."""

from __future__ import annotations

import numpy as np
import pytest

from photo_style.clustering import fit_clusters
from photo_style.inference import apply_profile
from photo_style.profile_store import StyleProfile, make_metadata
from photo_style.style_model import fit_cluster_transform, sample_pair_pixels


def _make_pair(rng: np.random.Generator, size=(96, 96)) -> tuple[np.ndarray, np.ndarray]:
    src = rng.integers(40, 200, size=(*size, 3), dtype=np.uint8)
    tgt = np.clip(src.astype(np.float32) * np.array([1.15, 1.0, 0.85]) + 6.0, 0, 255).astype(np.uint8)
    return src, tgt


def _single_cluster_profile(with_rf: bool) -> StyleProfile:
    rng = np.random.default_rng(0)
    pair_pixels = []
    for _ in range(2):
        src, tgt = _make_pair(rng)
        s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=500, rng=rng)
        pair_pixels.append((s_pix, t_pix))

    transform = fit_cluster_transform(
        pair_pixels, fit_rf=with_rf, rf_max_samples=1500, rf_n_estimators=10, seed=0
    )
    features = np.random.RandomState(0).randn(2, 30).astype(np.float32)
    cluster_model = fit_clusters(features, k=1)
    return StyleProfile(
        metadata=make_metadata("infer", photo_count=2, cluster_count=1, samples_per_pair=500, feature_max_dim=512),
        cluster_model=cluster_model,
        cluster_transforms={0: transform},
    )


def test_apply_profile_method_a_only() -> None:
    """Without an RF, apply_profile falls back to Reinhard + curves and still runs."""
    profile = _single_cluster_profile(with_rf=False)
    rng = np.random.default_rng(1)
    src = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
    styled, cluster_id = apply_profile(src, profile, use_rf=True)  # falls back since no RF
    assert styled.shape == src.shape
    assert styled.dtype == np.uint8
    assert cluster_id == 0


def test_apply_profile_method_b_when_rf_present() -> None:
    profile = _single_cluster_profile(with_rf=True)
    rng = np.random.default_rng(2)
    src = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
    styled_b, _ = apply_profile(src, profile, use_rf=True)
    styled_a, _ = apply_profile(src, profile, use_rf=False)
    # Both should produce uint8 RGB of the same shape.
    assert styled_a.shape == styled_b.shape == src.shape
    # Method A and B should generally differ pixel-wise.
    assert not np.array_equal(styled_a, styled_b)


def test_apply_profile_override_cluster() -> None:
    profile = _single_cluster_profile(with_rf=False)
    rng = np.random.default_rng(3)
    src = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
    styled, cid = apply_profile(src, profile, override_cluster=0)
    assert cid == 0
    with pytest.raises(KeyError):
        apply_profile(src, profile, override_cluster=99)
