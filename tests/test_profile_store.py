"""Tests for profile save/load roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from photo_style.clustering import fit_clusters
from photo_style.profile_store import (
    StyleProfile,
    list_profiles,
    load_profile,
    make_metadata,
    save_profile,
)
from photo_style.style_model import fit_cluster_transform, sample_pair_pixels


def _make_profile(name: str = "test") -> StyleProfile:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 8)).astype(np.float32)
    cluster_model = fit_clusters(features, k=3)

    transforms = {}
    for c in range(cluster_model.n_clusters):
        src = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
        tgt = np.clip(src.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=500, rng=rng)
        transforms[c] = fit_cluster_transform([(s_pix, t_pix)])

    metadata = make_metadata(
        name=name, photo_count=12, cluster_count=3, samples_per_pair=500, feature_max_dim=512
    )
    return StyleProfile(metadata=metadata, cluster_model=cluster_model, cluster_transforms=transforms)


def test_save_and_load_profile_roundtrip(tmp_path: Path) -> None:
    profile = _make_profile("alpha")
    pkl_path = save_profile(profile, profiles_dir=tmp_path)
    assert pkl_path.exists()
    assert (tmp_path / "alpha.json").exists()

    loaded = load_profile("alpha", profiles_dir=tmp_path)
    assert loaded.metadata.name == "alpha"
    assert loaded.metadata.photo_count == 12
    assert loaded.metadata.cluster_count == 3
    assert set(loaded.cluster_transforms.keys()) == set(profile.cluster_transforms.keys())
    # Reinhard stats and curves preserved exactly across the pickle.
    for c, t in profile.cluster_transforms.items():
        np.testing.assert_allclose(
            loaded.cluster_transforms[c].reinhard.tgt_mean, t.reinhard.tgt_mean
        )
        np.testing.assert_allclose(
            loaded.cluster_transforms[c].curves.L_poly.coef, t.curves.L_poly.coef
        )


def test_list_profiles_returns_saved_names(tmp_path: Path) -> None:
    save_profile(_make_profile("alpha"), profiles_dir=tmp_path)
    save_profile(_make_profile("beta"), profiles_dir=tmp_path)
    assert list_profiles(tmp_path) == ["alpha", "beta"]


def test_list_profiles_empty(tmp_path: Path) -> None:
    assert list_profiles(tmp_path) == []
