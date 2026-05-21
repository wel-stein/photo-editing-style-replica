"""Tests for K-Means clustering pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from photo_style.clustering import fit_clusters


def _three_blob_features(rng: np.random.Generator) -> np.ndarray:
    """Three well-separated 8-D blobs, 20 samples each."""
    centers = np.array([[0.0] * 8, [10.0] * 8, [-10.0] * 8])
    samples = []
    for c in centers:
        samples.append(c + rng.normal(scale=0.1, size=(20, 8)))
    return np.concatenate(samples, axis=0).astype(np.float32)


def test_fit_clusters_assigns_blobs_correctly() -> None:
    rng = np.random.default_rng(0)
    X = _three_blob_features(rng)
    model = fit_clusters(X, k=3)
    assert model.n_clusters == 3
    # Each blob (20 contiguous rows) should land in a single cluster.
    for start in (0, 20, 40):
        block = model.labels[start : start + 20]
        assert len(set(block.tolist())) == 1


def test_assign_handles_single_vector() -> None:
    rng = np.random.default_rng(1)
    X = _three_blob_features(rng)
    model = fit_clusters(X, k=3)
    new_point = np.array([10.0] * 8, dtype=np.float32)
    label = model.assign(new_point)
    assert label.shape == (1,)
    # Should land in the cluster that contains the [10, 10, ...] blob.
    assert label[0] == model.labels[20]


def test_fit_clusters_clamps_k_to_n_samples() -> None:
    """k larger than n_samples is clamped so this never crashes on tiny sets."""
    X = np.random.RandomState(0).randn(3, 8).astype(np.float32)
    model = fit_clusters(X, k=10)
    assert model.n_clusters == 3


def test_fit_clusters_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_clusters(np.zeros((0, 8), dtype=np.float32), k=3)
    with pytest.raises(ValueError):
        fit_clusters(np.zeros(8, dtype=np.float32), k=3)
