"""Tests for EXIF preservation on save (PIL-only, no extra deps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from photo_style.io_utils import save_image_rgb

# Standard EXIF tag IDs.
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110


def _write_jpg_with_exif(path: Path, make="NIKON CORPORATION", model="NIKON Z6") -> None:
    arr = np.full((64, 64, 3), 120, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    exif = img.getexif()
    exif[TAG_MAKE] = make
    exif[TAG_MODEL] = model
    img.save(path, exif=exif.tobytes(), quality=95)


def _read_make_model(path: Path) -> tuple[str, str]:
    with Image.open(path) as im:
        exif = im.getexif()
    return str(exif.get(TAG_MAKE, "")), str(exif.get(TAG_MODEL, ""))


def test_save_preserves_exif_from_jpeg_source(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _write_jpg_with_exif(src, make="NIKON CORPORATION", model="NIKON Z6")

    out_arr = np.full((64, 64, 3), 80, dtype=np.uint8)
    out = tmp_path / "out.jpg"
    save_image_rgb(out_arr, out, source_path=src)

    make, model = _read_make_model(out)
    assert "NIKON" in make
    assert "Z6" in model


def test_save_without_source_has_no_camera_exif(tmp_path: Path) -> None:
    out_arr = np.full((64, 64, 3), 80, dtype=np.uint8)
    out = tmp_path / "out.jpg"
    save_image_rgb(out_arr, out)  # no source_path

    make, model = _read_make_model(out)
    assert make == ""
    assert model == ""


def test_save_missing_source_is_graceful(tmp_path: Path) -> None:
    """A non-existent source path must not crash the save."""
    out_arr = np.full((32, 32, 3), 90, dtype=np.uint8)
    out = tmp_path / "out.jpg"
    save_image_rgb(out_arr, out, source_path=tmp_path / "does_not_exist.NEF")
    assert out.exists()


def test_save_png_output_ignores_exif(tmp_path: Path) -> None:
    """PNG output doesn't take the exif blob; save must still succeed."""
    src = tmp_path / "src.jpg"
    _write_jpg_with_exif(src)
    out_arr = np.full((32, 32, 3), 70, dtype=np.uint8)
    out = tmp_path / "out.png"
    save_image_rgb(out_arr, out, source_path=src)
    assert out.exists()
