"""Image I/O helpers. Phase 1: minimal RGB load/save via Pillow.

RAW support (rawpy) and EXIF preservation are added in later phases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an HxWx3 uint8 RGB ndarray."""
    path = Path(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def save_image_rgb(arr: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """Save an HxWx3 uint8 RGB ndarray to disk (JPEG quality applies to .jpg/.jpeg)."""
    path = Path(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=quality)
