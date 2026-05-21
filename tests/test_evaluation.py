"""Tests for ΔE2000 math and held-out validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from photo_style.clustering import fit_clusters
from photo_style.evaluation import (
    _opencv_lab_to_cie,
    evaluate_profile_on_pairs,
    mean_delta_e_2000,
)
from photo_style.profile_store import StyleProfile, make_metadata
from photo_style.style_model import (
    _rgb_to_lab,
    fit_cluster_transform,
    sample_pair_pixels,
)


def _write_jpg(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGB").save(path, quality=95)


def test_mean_delta_e_2000_identity_is_zero() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    lab = _rgb_to_lab(rgb)
    assert mean_delta_e_2000(lab, lab) < 1e-3


def test_mean_delta_e_2000_positive_for_different_images() -> None:
    rng = np.random.default_rng(0)
    rgb_a = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    rgb_b = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    de = mean_delta_e_2000(_rgb_to_lab(rgb_a), _rgb_to_lab(rgb_b))
    assert de > 1.0


def test_opencv_lab_to_cie_ranges() -> None:
    lab = np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    cie = _opencv_lab_to_cie(lab)
    assert abs(cie[0, 0, 0] - 100.0) < 1e-4
    assert abs(cie[0, 1, 0] - 0.0) < 1e-4
    assert abs(cie[0, 0, 1] - 127.0) < 1e-4
    assert abs(cie[0, 1, 1] - (-128.0)) < 1e-4


def _make_warm_pair(rng: np.random.Generator, size=(96, 96)) -> tuple[np.ndarray, np.ndarray]:
    src = rng.integers(40, 200, size=(*size, 3), dtype=np.uint8)
    edited = np.clip(src.astype(np.float32) * np.array([1.2, 1.0, 0.8]) + 8.0, 0, 255).astype(np.uint8)
    return src, edited


def _train_profile_on(pairs_rgb, rng) -> StyleProfile:
    feature_vectors = []
    pair_pixels = []
    for src, tgt in pairs_rgb:
        # Trivially produce a feature vector (LAB means) just to give clustering something.
        lab = _rgb_to_lab(src).reshape(-1, 3)
        feature_vectors.append(np.concatenate([lab.mean(axis=0), lab.std(axis=0), np.zeros(24)]).astype(np.float32))
        s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=500, rng=rng)
        pair_pixels.append((s_pix, t_pix))

    cluster_model = fit_clusters(np.stack(feature_vectors), k=1)
    transform = fit_cluster_transform(pair_pixels, fit_rf=True, rf_max_samples=2000, rf_n_estimators=10, seed=0)
    metadata = make_metadata("eval_test", photo_count=len(pairs_rgb), cluster_count=1, samples_per_pair=500, feature_max_dim=512)
    return StyleProfile(metadata=metadata, cluster_model=cluster_model, cluster_transforms={0: transform})


def test_evaluate_profile_returns_lower_de_than_identity(tmp_path: Path) -> None:
    """After training, ΔE on val pairs should drop below the identity baseline."""
    rng = np.random.default_rng(0)
    pairs_rgb = [_make_warm_pair(rng) for _ in range(6)]
    train_rgb, val_rgb = pairs_rgb[:4], pairs_rgb[4:]
    profile = _train_profile_on(train_rgb, rng)

    val_paths: list[tuple[Path, Path]] = []
    for i, (src, tgt) in enumerate(val_rgb):
        sp = tmp_path / f"val_src_{i}.jpg"
        tp = tmp_path / f"val_tgt_{i}.jpg"
        _write_jpg(sp, src)
        _write_jpg(tp, tgt)
        val_paths.append((sp, tp))

    metrics = evaluate_profile_on_pairs(profile, val_paths)
    assert metrics.n_pairs == 2
    assert metrics.identity > 0
    assert metrics.method_a < metrics.identity, "Method A should reduce ΔE vs identity"
    assert metrics.method_b is not None
    assert metrics.method_b < metrics.identity, "Method B should reduce ΔE vs identity"


def test_evaluate_profile_handles_empty_pair_list() -> None:
    rng = np.random.default_rng(0)
    pairs_rgb = [_make_warm_pair(rng) for _ in range(2)]
    profile = _train_profile_on(pairs_rgb, rng)
    metrics = evaluate_profile_on_pairs(profile, [])
    assert metrics.n_pairs == 0
