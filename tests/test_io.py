"""Tests for io_utils: image load + pair discovery against synthetic files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_style.io_utils import find_pairs, load_image_rgb, save_image_rgb


def _write_solid_jpg(path: Path, color=(128, 128, 128), size=(32, 32)) -> None:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=95)


def test_load_image_rgb_returns_uint8_rgb(tmp_path: Path) -> None:
    p = tmp_path / "img.jpg"
    _write_solid_jpg(p, color=(200, 100, 50))
    arr = load_image_rgb(p)
    assert arr.shape == (32, 32, 3)
    assert arr.dtype == np.uint8


def test_save_image_rgb_roundtrip(tmp_path: Path) -> None:
    arr = np.full((16, 16, 3), 80, dtype=np.uint8)
    p = tmp_path / "out.png"
    save_image_rgb(arr, p)
    loaded = load_image_rgb(p)
    assert loaded.shape == arr.shape
    assert (loaded == 80).all()


def test_find_pairs_matches_by_stem(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    for name in ("shot1.jpg", "shot2.jpg"):
        _write_solid_jpg(originals / name)
        _write_solid_jpg(edited / name)

    result = find_pairs(originals, edited)
    assert result.num_pairs == 2
    assert result.unmatched_originals == []
    assert result.unmatched_edited == []


def test_find_pairs_reports_unmatched(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    _write_solid_jpg(originals / "shot1.jpg")
    _write_solid_jpg(originals / "shot2.jpg")
    _write_solid_jpg(edited / "shot1.jpg")
    _write_solid_jpg(edited / "shot3.jpg")

    result = find_pairs(originals, edited)
    assert result.num_pairs == 1
    assert [p.name for p in result.unmatched_originals] == ["shot2.jpg"]
    assert [p.name for p in result.unmatched_edited] == ["shot3.jpg"]


def test_find_pairs_cross_extension(tmp_path: Path) -> None:
    """NEF in originals/ should pair with JPG in edited/ by stem alone."""
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    # We don't decode the NEF here; pair discovery only cares about extension + stem.
    (originals / "DSC_001.NEF").write_bytes(b"")
    _write_solid_jpg(edited / "DSC_001.jpg")

    result = find_pairs(originals, edited)
    assert result.num_pairs == 1
    src, tgt = result.pairs[0]
    assert src.suffix == ".NEF"
    assert tgt.suffix == ".jpg"


def test_find_pairs_case_insensitive(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    _write_solid_jpg(originals / "Shot1.jpg")
    _write_solid_jpg(edited / "SHOT1.JPG")

    result = find_pairs(originals, edited)
    assert result.num_pairs == 1


def test_find_pairs_prefers_raw_when_duplicate_stem(tmp_path: Path) -> None:
    """If originals/ has both IMG_001.NEF and IMG_001.jpg, the RAW wins."""
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    (originals / "IMG_001.NEF").write_bytes(b"")
    _write_solid_jpg(originals / "IMG_001.jpg")
    _write_solid_jpg(edited / "IMG_001.jpg")

    result = find_pairs(originals, edited)
    assert result.num_pairs == 1
    assert result.pairs[0][0].suffix == ".NEF"


def test_find_pairs_handles_missing_folders(tmp_path: Path) -> None:
    result = find_pairs(tmp_path / "nope_a", tmp_path / "nope_b")
    assert result.num_pairs == 0
    assert result.unmatched_originals == []
    assert result.unmatched_edited == []


def test_find_pairs_ignores_non_image_files(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    _write_solid_jpg(originals / "shot1.jpg")
    _write_solid_jpg(edited / "shot1.jpg")
    (originals / "notes.txt").write_text("ignore me")
    (edited / "thumbs.db").write_bytes(b"")

    result = find_pairs(originals, edited)
    assert result.num_pairs == 1
    assert result.unmatched_originals == []
    assert result.unmatched_edited == []
