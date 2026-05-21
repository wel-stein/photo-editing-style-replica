"""Validation metrics: CIEDE2000 between predicted and actual edits.

OpenCV LAB is on a 0-255 per-channel scale; CIE LAB used by `skimage` is
L=[0,100], a/b ~[-128,127]. We convert before measuring ΔE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from skimage.color import deltaE_ciede2000

from .io_utils import load_image_rgb
from .profile_store import StyleProfile
from .style_model import _rgb_to_lab, apply_cluster_transform

EVAL_MAX_DIM = 512  # downsample held-out pairs to this longest edge for speed


def _opencv_lab_to_cie(lab_cv: np.ndarray) -> np.ndarray:
    out = lab_cv.astype(np.float32).copy()
    out[..., 0] *= 100.0 / 255.0
    out[..., 1] -= 128.0
    out[..., 2] -= 128.0
    return out


def mean_delta_e_2000(lab_a_cv: np.ndarray, lab_b_cv: np.ndarray) -> float:
    """Mean CIEDE2000 between two LAB images on OpenCV's 0-255 scale."""
    a = _opencv_lab_to_cie(lab_a_cv)
    b = _opencv_lab_to_cie(lab_b_cv)
    return float(deltaE_ciede2000(a, b).mean())


def _resize_to_match(src_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """Resize target to source dimensions if they differ slightly."""
    if tgt_rgb.shape[:2] == src_rgb.shape[:2]:
        return tgt_rgb
    h, w = src_rgb.shape[:2]
    return cv2.resize(tgt_rgb, (w, h), interpolation=cv2.INTER_AREA)


def _downsample(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return rgb
    scale = max_dim / longest
    return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class ValidationMetrics:
    """Mean CIEDE2000 across the validation set under each method."""

    n_pairs: int
    identity: float           # ΔE between src and tgt with no transform
    method_a: float           # ΔE after Reinhard + tonal curves
    method_b: float | None    # ΔE after RF (None if no RF was trained)

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "identity": self.identity,
            "method_a": self.method_a,
            "method_b": self.method_b,
        }


def evaluate_profile_on_pairs(
    profile: StyleProfile,
    pairs: list[tuple[Path, Path]],
    *,
    max_dim: int = EVAL_MAX_DIM,
) -> ValidationMetrics:
    """Run identity / Method A / Method B over the held-out pairs and average ΔE."""
    if not pairs:
        return ValidationMetrics(n_pairs=0, identity=float("nan"), method_a=float("nan"), method_b=None)

    has_rf = any(t.pixel_rf is not None for t in profile.cluster_transforms.values())

    identity_vals: list[float] = []
    method_a_vals: list[float] = []
    method_b_vals: list[float] = []

    from .features import extract_features  # local import to keep module graph light

    for src_path, tgt_path in pairs:
        src_rgb = _downsample(load_image_rgb(src_path), max_dim)
        tgt_rgb = _resize_to_match(src_rgb, _downsample(load_image_rgb(tgt_path), max_dim))

        identity_vals.append(mean_delta_e_2000(_rgb_to_lab(src_rgb), _rgb_to_lab(tgt_rgb)))

        feature_vec = extract_features(src_rgb).to_vector()
        cluster_id = int(profile.cluster_model.assign(feature_vec)[0])
        transform = profile.cluster_transforms.get(cluster_id)
        if transform is None:
            logger.warning("Validation pair assigned to empty cluster {}; skipping", cluster_id)
            continue

        styled_a = apply_cluster_transform(src_rgb, transform, use_rf=False)
        method_a_vals.append(mean_delta_e_2000(_rgb_to_lab(styled_a), _rgb_to_lab(tgt_rgb)))

        if has_rf and transform.pixel_rf is not None:
            styled_b = apply_cluster_transform(src_rgb, transform, use_rf=True)
            method_b_vals.append(mean_delta_e_2000(_rgb_to_lab(styled_b), _rgb_to_lab(tgt_rgb)))

    return ValidationMetrics(
        n_pairs=len(pairs),
        identity=float(np.mean(identity_vals)) if identity_vals else float("nan"),
        method_a=float(np.mean(method_a_vals)) if method_a_vals else float("nan"),
        method_b=float(np.mean(method_b_vals)) if method_b_vals else None,
    )
