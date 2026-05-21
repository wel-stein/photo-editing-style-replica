"""Style models. Phase 1: Reinhard color transfer in LAB only.

Polynomial tone curves and the Random Forest pixel regressor land in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReinhardStats:
    """Per-channel mean and std in LAB for source and target distributions."""

    src_mean: np.ndarray  # shape (3,)
    src_std: np.ndarray
    tgt_mean: np.ndarray
    tgt_std: np.ndarray


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 LAB using OpenCV's D65 sRGB conversion."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab


def _lab_to_rgb(lab_f32: np.ndarray) -> np.ndarray:
    """float32 LAB (OpenCV's 0-255 range) -> uint8 RGB."""
    lab = np.clip(lab_f32, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_reinhard(src_rgb: np.ndarray, tgt_rgb: np.ndarray) -> ReinhardStats:
    """Compute Reinhard mean/std stats in LAB for a single source/target pair."""
    src_lab = _rgb_to_lab(src_rgb).reshape(-1, 3)
    tgt_lab = _rgb_to_lab(tgt_rgb).reshape(-1, 3)
    return ReinhardStats(
        src_mean=src_lab.mean(axis=0),
        src_std=src_lab.std(axis=0) + 1e-6,
        tgt_mean=tgt_lab.mean(axis=0),
        tgt_std=tgt_lab.std(axis=0) + 1e-6,
    )


def apply_reinhard(rgb_u8: np.ndarray, stats: ReinhardStats) -> np.ndarray:
    """Apply a Reinhard transfer (LAB mean/std rescaling) to an RGB image."""
    lab = _rgb_to_lab(rgb_u8)
    centered = lab - stats.src_mean
    scaled = centered * (stats.tgt_std / stats.src_std)
    out = scaled + stats.tgt_mean
    return _lab_to_rgb(out)
