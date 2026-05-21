"""Style models: Reinhard color transfer, per-channel polynomial tone curves,
and per-cluster transforms that stack them.

All math runs in OpenCV's LAB (channels scaled 0-255). The Random Forest
pixel regressor (Method B) lands in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.polynomial import Polynomial

LAB_CHANNELS = ("L", "a", "b")
DEFAULT_CURVE_DEGREE = 3
DEFAULT_PIXEL_SAMPLE_MAX_DIM = 1024


@dataclass
class ReinhardStats:
    """Per-channel mean and std in LAB for source and target distributions."""

    src_mean: np.ndarray  # shape (3,)
    src_std: np.ndarray
    tgt_mean: np.ndarray
    tgt_std: np.ndarray


@dataclass
class TonalCurves:
    """Per-channel polynomial tone curves (one Polynomial per LAB channel).

    Each `Polynomial` carries its own input-domain mapping internally, so fits
    are well-conditioned regardless of how narrow the channel's range is.
    """

    L_poly: Polynomial
    a_poly: Polynomial
    b_poly: Polynomial
    degree: int = DEFAULT_CURVE_DEGREE


@dataclass
class ClusterTransform:
    """Style transform for one cluster: Reinhard + residual polynomial curves."""

    reinhard: ReinhardStats
    curves: TonalCurves


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 LAB (OpenCV 0-255 scale)."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _lab_to_rgb(lab_f32: np.ndarray) -> np.ndarray:
    """float32 LAB -> uint8 RGB."""
    lab = np.clip(lab_f32, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_reinhard(src_rgb: np.ndarray, tgt_rgb: np.ndarray) -> ReinhardStats:
    """Compute Reinhard mean/std stats in LAB for a single source/target pair."""
    src_lab = _rgb_to_lab(src_rgb).reshape(-1, 3)
    tgt_lab = _rgb_to_lab(tgt_rgb).reshape(-1, 3)
    return _reinhard_from_lab_pixels(src_lab, tgt_lab)


def _reinhard_from_lab_pixels(src_lab: np.ndarray, tgt_lab: np.ndarray) -> ReinhardStats:
    return ReinhardStats(
        src_mean=src_lab.mean(axis=0),
        src_std=src_lab.std(axis=0) + 1e-6,
        tgt_mean=tgt_lab.mean(axis=0),
        tgt_std=tgt_lab.std(axis=0) + 1e-6,
    )


def _apply_reinhard_lab(lab_pixels: np.ndarray, stats: ReinhardStats) -> np.ndarray:
    centered = lab_pixels - stats.src_mean
    scaled = centered * (stats.tgt_std / stats.src_std)
    return scaled + stats.tgt_mean


def apply_reinhard(rgb_u8: np.ndarray, stats: ReinhardStats) -> np.ndarray:
    """Apply a Reinhard transfer (LAB mean/std rescaling) to an RGB image."""
    lab = _rgb_to_lab(rgb_u8)
    out = _apply_reinhard_lab(lab, stats)
    return _lab_to_rgb(out)


def sample_pair_pixels(
    src_rgb: np.ndarray,
    tgt_rgb: np.ndarray,
    n_samples: int,
    max_dim: int = DEFAULT_PIXEL_SAMPLE_MAX_DIM,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample co-located LAB pixels from a pair.

    Resizes the target to the source's dimensions (NEFs and their JPEG edits
    sometimes differ by a few pixels), then downsamples both to `max_dim`
    longest edge before random sampling so this stays cheap on full-res RAWs.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = src_rgb.shape[:2]
    if tgt_rgb.shape[:2] != (h, w):
        tgt_rgb = cv2.resize(tgt_rgb, (w, h), interpolation=cv2.INTER_AREA)

    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        src_rgb = cv2.resize(src_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tgt_rgb = cv2.resize(tgt_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    src_lab = _rgb_to_lab(src_rgb).reshape(-1, 3)
    tgt_lab = _rgb_to_lab(tgt_rgb).reshape(-1, 3)

    n = min(n_samples, len(src_lab))
    idx = rng.choice(len(src_lab), size=n, replace=False)
    return src_lab[idx], tgt_lab[idx]


def fit_cluster_transform(
    pair_pixels: list[tuple[np.ndarray, np.ndarray]],
    degree: int = DEFAULT_CURVE_DEGREE,
) -> ClusterTransform:
    """Fit Reinhard stats + per-channel residual polynomial curves for one cluster.

    `pair_pixels` is a list of (src_lab_samples, tgt_lab_samples) from each pair
    assigned to this cluster. The polynomial is fit on the *residual* after the
    Reinhard step, so its job is just to capture nonlinear tonal shape.
    """
    if not pair_pixels:
        raise ValueError("Cannot fit a cluster transform with zero pairs")

    all_src = np.concatenate([s for s, _ in pair_pixels], axis=0)
    all_tgt = np.concatenate([t for _, t in pair_pixels], axis=0)

    reinhard = _reinhard_from_lab_pixels(all_src, all_tgt)
    intermediate = _apply_reinhard_lab(all_src, reinhard)

    L_poly = Polynomial.fit(intermediate[:, 0], all_tgt[:, 0], deg=degree)
    a_poly = Polynomial.fit(intermediate[:, 1], all_tgt[:, 1], deg=degree)
    b_poly = Polynomial.fit(intermediate[:, 2], all_tgt[:, 2], deg=degree)
    curves = TonalCurves(L_poly=L_poly, a_poly=a_poly, b_poly=b_poly, degree=degree)

    return ClusterTransform(reinhard=reinhard, curves=curves)


def _apply_curves_lab(lab_pixels: np.ndarray, curves: TonalCurves) -> np.ndarray:
    out = np.empty_like(lab_pixels)
    out[..., 0] = curves.L_poly(lab_pixels[..., 0])
    out[..., 1] = curves.a_poly(lab_pixels[..., 1])
    out[..., 2] = curves.b_poly(lab_pixels[..., 2])
    return out


def apply_cluster_transform(rgb_u8: np.ndarray, transform: ClusterTransform) -> np.ndarray:
    """Apply Reinhard then per-channel polynomial curves to an RGB image."""
    lab = _rgb_to_lab(rgb_u8)
    lab = _apply_reinhard_lab(lab, transform.reinhard)
    lab = _apply_curves_lab(lab, transform.curves)
    return _lab_to_rgb(lab)
