"""Tests for .cube LUT generation and writing."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from photo_style.clustering import fit_clusters
from photo_style.lut_export import (
    DEFAULT_LUT_SIZE,
    export_profile_lut,
    generate_average_lut_array,
    generate_lut_array,
    write_cube,
)
from photo_style.profile_store import StyleProfile, make_metadata
from photo_style.style_model import fit_cluster_transform, sample_pair_pixels


def _make_pair(rng: np.random.Generator, size=(96, 96)) -> tuple[np.ndarray, np.ndarray]:
    src = rng.integers(40, 200, size=(*size, 3), dtype=np.uint8)
    tgt = np.clip(src.astype(np.float32) * np.array([1.15, 1.0, 0.85]) + 6.0, 0, 255).astype(np.uint8)
    return src, tgt


def _two_cluster_profile(with_rf: bool = False) -> StyleProfile:
    rng = np.random.default_rng(0)
    transforms = {}
    for c in range(2):
        pair_pixels = []
        for _ in range(2):
            src, tgt = _make_pair(rng)
            s_pix, t_pix = sample_pair_pixels(src, tgt, n_samples=400, rng=rng)
            pair_pixels.append((s_pix, t_pix))
        transforms[c] = fit_cluster_transform(
            pair_pixels, fit_rf=with_rf, rf_max_samples=600, rf_n_estimators=6, seed=c
        )
    features = np.random.RandomState(0).randn(4, 30).astype(np.float32)
    cluster_model = fit_clusters(features, k=2)
    return StyleProfile(
        metadata=make_metadata("lut_test", photo_count=4, cluster_count=2, samples_per_pair=400, feature_max_dim=512),
        cluster_model=cluster_model,
        cluster_transforms=transforms,
    )


def test_generate_lut_array_shape_and_range() -> None:
    profile = _two_cluster_profile()
    lut = generate_lut_array(profile.cluster_transforms[0], size=DEFAULT_LUT_SIZE, use_rf=False)
    assert lut.shape == (DEFAULT_LUT_SIZE, DEFAULT_LUT_SIZE, DEFAULT_LUT_SIZE, 3)
    assert lut.dtype == np.float32
    # OpenCV uint8 path keeps values in [0, 1] after the /255 normalization.
    assert lut.min() >= 0.0 and lut.max() <= 1.0


def test_generate_lut_array_smaller_size() -> None:
    profile = _two_cluster_profile()
    lut = generate_lut_array(profile.cluster_transforms[0], size=9, use_rf=False)
    assert lut.shape == (9, 9, 9, 3)


def test_generate_lut_array_rejects_size_lt_2() -> None:
    profile = _two_cluster_profile()
    with pytest.raises(ValueError):
        generate_lut_array(profile.cluster_transforms[0], size=1)


def test_generate_average_lut_array_shape() -> None:
    profile = _two_cluster_profile()
    avg = generate_average_lut_array(profile, size=9, use_rf=False)
    assert avg.shape == (9, 9, 9, 3)
    assert 0.0 <= avg.min() and avg.max() <= 1.0


def test_write_cube_produces_parseable_file(tmp_path: Path) -> None:
    profile = _two_cluster_profile()
    lut = generate_lut_array(profile.cluster_transforms[0], size=9, use_rf=False)
    out = tmp_path / "out.cube"
    write_cube(lut, out, title="unit_test")

    text = out.read_text()
    assert 'TITLE "unit_test"' in text
    assert "LUT_3D_SIZE 9" in text
    assert "DOMAIN_MIN 0.0 0.0 0.0" in text
    assert "DOMAIN_MAX 1.0 1.0 1.0" in text

    data_lines = [
        line for line in text.splitlines()
        if line and re.match(r"^[\d\.\s]+$", line) and not line.startswith("LUT_3D_SIZE")
    ]
    # 9^3 = 729 data rows
    assert len(data_lines) == 9 ** 3
    for line in data_lines[:5]:
        parts = line.split()
        assert len(parts) == 3
        for p in parts:
            v = float(p)
            assert 0.0 <= v <= 1.0


def test_write_cube_byte_order_is_r_fastest(tmp_path: Path) -> None:
    """The .cube spec requires R to vary fastest; verify by constructing a
    distinguishable LUT and reading back the first ramp."""
    size = 4
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    # Encode r index into the red channel; g and b indices into the others.
    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                lut[ri, gi, bi] = [ri / (size - 1), gi / (size - 1), bi / (size - 1)]

    out = tmp_path / "ramp.cube"
    write_cube(lut, out)
    lines = [
        l for l in out.read_text().splitlines()
        if re.match(r"^[\d\.]+\s+[\d\.]+\s+[\d\.]+$", l)
    ]
    # First `size` lines should be a pure red ramp at (g=0, b=0).
    first_ramp = [tuple(map(float, l.split())) for l in lines[:size]]
    for i, (r, g, b) in enumerate(first_ramp):
        assert abs(r - i / (size - 1)) < 1e-5
        assert g == 0.0
        assert b == 0.0


def test_export_profile_lut_average(tmp_path: Path) -> None:
    profile = _two_cluster_profile()
    out = export_profile_lut(profile, tmp_path / "avg.cube", cluster="average", size=9, use_rf=False)
    assert out.exists()
    assert 'TITLE "lut_test_average"' in out.read_text()


def test_export_profile_lut_specific_cluster(tmp_path: Path) -> None:
    profile = _two_cluster_profile()
    out = export_profile_lut(profile, tmp_path / "c1.cube", cluster=1, size=9, use_rf=False)
    assert 'TITLE "lut_test_cluster1"' in out.read_text()


def test_export_profile_lut_unknown_cluster_raises(tmp_path: Path) -> None:
    profile = _two_cluster_profile()
    with pytest.raises(KeyError):
        export_profile_lut(profile, tmp_path / "x.cube", cluster=99, size=9)
