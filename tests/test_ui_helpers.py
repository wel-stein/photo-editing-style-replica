"""Tests for the pure data-logic helpers behind the Gradio UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from photo_style.ui import (
    _cluster_choices_for,
    helper_apply,
    helper_apply_batch,
    helper_export_lut,
    helper_match_pairs,
    helper_train,
)


def _write_jpg(path: Path, color=(128, 128, 128), size=(64, 64)) -> None:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=95)


def _seed_synthetic_pairs(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    originals = tmp_path / "originals"
    edited = tmp_path / "edited"
    originals.mkdir()
    edited.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        src = rng.integers(40, 200, size=(96, 96, 3), dtype=np.uint8)
        tgt = np.clip(src.astype(np.float32) * np.array([1.15, 1.0, 0.85]) + 6, 0, 255).astype(np.uint8)
        Image.fromarray(src, "RGB").save(originals / f"shot_{i:02d}.jpg", quality=95)
        Image.fromarray(tgt, "RGB").save(edited / f"shot_{i:02d}.jpg", quality=95)
    return originals, edited


def test_helper_match_pairs_reports_counts(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=3)
    msg, unmatched = helper_match_pairs(str(originals), str(edited))
    assert "Matched 3 pair(s)" in msg
    assert unmatched == []


def test_helper_match_pairs_lists_unmatched(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=2)
    _write_jpg(originals / "extra.jpg")
    msg, unmatched = helper_match_pairs(str(originals), str(edited))
    assert "Matched 2 pair(s)" in msg
    assert any("extra.jpg" in u for u in unmatched)


def test_helper_match_pairs_handles_empty_input() -> None:
    msg, unmatched = helper_match_pairs("", "")
    assert "Please enter both folder paths." in msg
    assert unmatched == []


def test_helper_train_end_to_end(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=6)
    profiles_dir = tmp_path / "profiles"
    cache_path = tmp_path / "cache.pkl"

    progress_calls: list[tuple[float, str]] = []

    def _progress(frac: float, desc: str) -> None:
        progress_calls.append((frac, desc))

    status, feats, labels = helper_train(
        str(originals),
        str(edited),
        k=2,
        samples_per_pair=500,
        val_fraction=0.25,
        use_rf=False,  # skip RF to keep the test fast
        profile_name="ui_test",
        progress=_progress,
        profiles_dir=profiles_dir,
        cache_path=cache_path,
    )

    assert "Saved 'ui_test'" in status
    assert (profiles_dir / "ui_test.pkl").exists()
    assert (profiles_dir / "ui_test.json").exists()
    assert feats is not None and labels is not None
    assert labels.shape[0] == feats.shape[0]
    assert progress_calls, "progress callback was never invoked"
    assert progress_calls[-1][0] == 1.0


def test_helper_train_rejects_empty_name(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=3)
    status, feats, labels = helper_train(
        str(originals),
        str(edited),
        k=2,
        samples_per_pair=500,
        val_fraction=0.0,
        use_rf=False,
        profile_name="   ",
    )
    assert "profile name" in status.lower()
    assert feats is None and labels is None


def test_helper_train_handles_no_pairs(tmp_path: Path) -> None:
    (tmp_path / "originals").mkdir()
    (tmp_path / "edited").mkdir()
    status, feats, labels = helper_train(
        str(tmp_path / "originals"),
        str(tmp_path / "edited"),
        k=2,
        samples_per_pair=500,
        val_fraction=0.0,
        use_rf=False,
        profile_name="empty",
    )
    assert "No matched pairs" in status
    assert feats is None


def test_helper_apply_styles_image(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=4)
    profiles_dir = tmp_path / "profiles"
    helper_train(
        str(originals), str(edited), k=2, samples_per_pair=300, val_fraction=0.0,
        use_rf=False, profile_name="apply_test", profiles_dir=profiles_dir,
        cache_path=tmp_path / "cache.pkl",
    )

    src_path = next(iter(originals.glob("*.jpg")))
    before, after, dl, status = helper_apply(
        "apply_test", str(src_path), use_rf=False, profiles_dir=profiles_dir
    )
    assert before is not None and after is not None
    assert before.shape == after.shape
    assert before.dtype == after.dtype == np.uint8
    assert dl is not None and Path(dl).exists()
    assert "apply_test" in status


def test_helper_apply_missing_inputs() -> None:
    before, after, dl, status = helper_apply("", None, use_rf=False)
    assert before is None and after is None and dl is None
    assert "profile" in status.lower()


def test_helper_apply_batch_produces_zip(tmp_path: Path) -> None:
    import zipfile

    originals, edited = _seed_synthetic_pairs(tmp_path, n=4)
    profiles_dir = tmp_path / "profiles"
    helper_train(
        str(originals), str(edited), k=2, samples_per_pair=300, val_fraction=0.0,
        use_rf=False, profile_name="batch_test", profiles_dir=profiles_dir,
        cache_path=tmp_path / "cache.pkl",
    )

    inputs = [str(p) for p in sorted(originals.glob("*.jpg"))]
    zip_path, status = helper_apply_batch(
        "batch_test", inputs, use_rf=False, profiles_dir=profiles_dir
    )
    assert zip_path is not None and Path(zip_path).exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert len(names) == len(inputs)
    assert all(n.endswith("_styled.jpg") for n in names)
    assert f"{len(inputs)}/{len(inputs)}" in status


def test_helper_apply_batch_missing_inputs() -> None:
    zip_path, status = helper_apply_batch("", None, use_rf=False)
    assert zip_path is None
    assert "profile" in status.lower()

    zip_path, status = helper_apply_batch("some_profile", [], use_rf=False)
    assert zip_path is None
    assert "upload" in status.lower()


def test_helper_export_lut_average(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=4)
    profiles_dir = tmp_path / "profiles"
    helper_train(
        str(originals), str(edited), k=2, samples_per_pair=300, val_fraction=0.0,
        use_rf=False, profile_name="lut_ui", profiles_dir=profiles_dir,
        cache_path=tmp_path / "cache.pkl",
    )

    path, status = helper_export_lut(
        "lut_ui", "average", size=9, use_rf=False, profiles_dir=profiles_dir
    )
    assert path is not None and Path(path).exists()
    assert Path(path).suffix == ".cube"
    assert "9x9x9" in status


def test_cluster_choices_for_known_profile(tmp_path: Path) -> None:
    originals, edited = _seed_synthetic_pairs(tmp_path, n=4)
    profiles_dir = tmp_path / "profiles"
    helper_train(
        str(originals), str(edited), k=2, samples_per_pair=300, val_fraction=0.0,
        use_rf=False, profile_name="cc_test", profiles_dir=profiles_dir,
        cache_path=tmp_path / "cache.pkl",
    )
    choices = _cluster_choices_for("cc_test", profiles_dir=profiles_dir)
    assert choices[0] == "average"
    assert set(choices[1:]) == {"0", "1"}


def test_cluster_choices_for_unknown_profile(tmp_path: Path) -> None:
    choices = _cluster_choices_for("nonexistent", profiles_dir=tmp_path)
    assert choices == ["average"]
