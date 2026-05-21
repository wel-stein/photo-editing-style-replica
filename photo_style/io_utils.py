"""Image I/O helpers. Phase 1: RGB load/save via Pillow + RAW load via rawpy.

EXIF preservation lands in a later phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf", ".pef"}


def _load_raw_rgb(path: Path) -> np.ndarray:
    """Demosaic a RAW file to 16-bit linear, return as 8-bit sRGB for processing."""
    import rawpy  # imported lazily so non-RAW workflows don't need it at import time

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.222, 4.5),  # standard sRGB-ish gamma
        )
    return (rgb16 / 257).astype(np.uint8)  # 65535 / 255 ≈ 257


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an HxWx3 uint8 RGB ndarray.

    Standard formats go through Pillow; RAW formats (NEF, CR2, ARW, DNG, ...)
    are demosaiced via rawpy and converted to sRGB.
    """
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw_rgb(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def save_image_rgb(arr: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """Save an HxWx3 uint8 RGB ndarray to disk (JPEG quality applies to .jpg/.jpeg)."""
    path = Path(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=quality)
