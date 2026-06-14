"""Image I/O helpers: RGB load/save via Pillow, RAW load via rawpy, pair discovery.

Output images carry EXIF from their source when possible (camera, lens,
exposure, date), per the project's "don't strip EXIF" rule.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf", ".pef"}
STANDARD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_EXTENSIONS = STANDARD_IMAGE_EXTENSIONS | RAW_EXTENSIONS
# Output formats whose Pillow writer accepts an `exif=` blob.
EXIF_WRITABLE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


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


def _extract_exif(source_path: Path) -> bytes | None:
    """Return raw EXIF bytes from a source image, or None if unavailable.

    Standard images are read directly by Pillow. RAW files expose EXIF via
    their embedded JPEG preview (which rawpy can extract), so we read the
    preview's EXIF rather than the harder-to-parse raw container.
    """
    source_path = Path(source_path)
    try:
        if source_path.suffix.lower() in RAW_EXTENSIONS:
            import rawpy

            with rawpy.imread(str(source_path)) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(thumb.data)) as preview:
                    return preview.info.get("exif")
            return None
        with Image.open(source_path) as im:
            return im.info.get("exif")
    except Exception as exc:  # noqa: BLE001 - EXIF is best-effort, never fatal
        logger.debug("Could not extract EXIF from {}: {}", source_path, exc)
        return None


def save_image_rgb(
    arr: np.ndarray,
    path: str | Path,
    quality: int = 95,
    source_path: str | Path | None = None,
) -> None:
    """Save an HxWx3 uint8 RGB ndarray to disk (JPEG quality applies to .jpg/.jpeg).

    When `source_path` is given and the output format supports it, EXIF metadata
    from the source image is copied to the output so camera/lens/exposure/date
    survive. Pixel data is never rotated or cropped, so an orientation tag
    carried over keeps the same display semantics as the source.
    """
    path = Path(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    exif_bytes: bytes | None = None
    if source_path is not None and path.suffix.lower() in EXIF_WRITABLE_EXTENSIONS:
        exif_bytes = _extract_exif(Path(source_path))

    img = Image.fromarray(arr, mode="RGB")
    if exif_bytes:
        img.save(path, quality=quality, exif=exif_bytes)
    else:
        img.save(path, quality=quality)


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
