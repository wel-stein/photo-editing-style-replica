"""Image I/O helpers: RGB load/save via Pillow, RAW load via rawpy, pair discovery.

EXIF preservation lands in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf", ".pef"}
STANDARD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_EXTENSIONS = STANDARD_IMAGE_EXTENSIONS | RAW_EXTENSIONS


@dataclass
class PairResult:
    """Outcome of matching originals to edits by filename stem."""

    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    unmatched_originals: list[Path] = field(default_factory=list)
    unmatched_edited: list[Path] = field(default_factory=list)

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)


def _load_raw_rgb(path: Path) -> np.ndarray:
    """Demosaic a RAW file to 16-bit linear, return as 8-bit sRGB for processing."""
    import rawpy  # imported lazily so non-RAW workflows don't need it at import time

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
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


def _index_images(folder: Path) -> dict[str, Path]:
    """Map lowercased filename stem -> path for image files in `folder` (non-recursive).

    If multiple files share a stem (e.g. IMG_001.NEF and IMG_001.JPG in the same
    folder), RAW files win — they're the better "original" for style learning.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = p.stem.lower()
        existing = out.get(stem)
        if existing is None or (
            existing.suffix.lower() not in RAW_EXTENSIONS
            and p.suffix.lower() in RAW_EXTENSIONS
        ):
            out[stem] = p
    return out


def find_pairs(originals_dir: str | Path, edited_dir: str | Path) -> PairResult:
    """Match originals to edits by case-insensitive filename stem.

    Returns matched pairs plus any unmatched files on either side so the UI can
    surface them to the user.
    """
    src = _index_images(Path(originals_dir))
    tgt = _index_images(Path(edited_dir))

    common = sorted(set(src) & set(tgt))
    only_src = sorted(set(src) - set(tgt))
    only_tgt = sorted(set(tgt) - set(src))

    return PairResult(
        pairs=[(src[s], tgt[s]) for s in common],
        unmatched_originals=[src[s] for s in only_src],
        unmatched_edited=[tgt[s] for s in only_tgt],
    )
