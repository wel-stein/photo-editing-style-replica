"""Style models: Reinhard color transfer, per-channel polynomial tone curves,
per-cluster transforms that stack them, and an optional Random Forest pixel
regressor (Method B).

All math runs in OpenCV's LAB (channels scaled 0-255).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.ensemble import RandomForestRegressor

LAB_CHANNELS = ("L", "a", "b")
DEFAULT_CURVE_DEGREE = 3
DEFAULT_PIXEL_SAMPLE_MAX_DIM = 1024
DEFAULT_RF_N_ESTIMATORS = 40
DEFAULT_RF_MAX_DEPTH = 10
DEFAULT_RF_MAX_SAMPLES = 80_000
# Side length of the LUT baked from the RF at train time for fast inference.
DEFAULT_RF_LUT_SIZE = 33


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
    """Style transform for one cluster.

    Always carries Method A (Reinhard + residual polynomial curves).
    `pixel_rf` is the optional Method B Random Forest pixel regressor (LAB
    in -> LAB out). When present, it overrides Method A at inference.

    `rf_lut` is the RF baked onto an RGB grid at train time (shape
    (N, N, N, 3), values in [0, 1]). When present it's applied via fast
    trilinear interpolation instead of running the forest per pixel,
    which turns a 30-60s full-res apply into a sub-second one with no
    visible quality change. `rf_lut_size` records N.
    """

    reinhard: ReinhardStats
    curves: TonalCurves
    pixel_rf: RandomForestRegressor | None = None
    rf_lut: np.ndarray | None = None
    rf_lut_size: int = 0


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
    *,
    degree: int = DEFAULT_CURVE_DEGREE,
    fit_rf: bool = False,
    rf_n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    rf_max_depth: int = DEFAULT_RF_MAX_DEPTH,
    rf_max_samples: int = DEFAULT_RF_MAX_SAMPLES,
    seed: int = 0,
) -> ClusterTransform:
    """Fit Reinhard stats + per-channel residual polynomial curves for one cluster.

    `pair_pixels` is a list of (src_lab_samples, tgt_lab_samples) from each pair
    assigned to this cluster. The polynomial is fit on the *residual* after the
    Reinhard step, so its job is just to capture nonlinear tonal shape.

    Set `fit_rf=True` to also train a Random Forest pixel regressor (Method B).
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

    pixel_rf = None
    rf_lut = None
    rf_lut_size = 0
    if fit_rf:
        pixel_rf = fit_pixel_rf(
            all_src,
            all_tgt,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_samples=rf_max_samples,
            seed=seed,
        )
        rf_lut = bake_rf_lut(pixel_rf, DEFAULT_RF_LUT_SIZE)
        rf_lut_size = DEFAULT_RF_LUT_SIZE

    return ClusterTransform(
        reinhard=reinhard,
        curves=curves,
        pixel_rf=pixel_rf,
        rf_lut=rf_lut,
        rf_lut_size=rf_lut_size,
    )


def bake_rf_lut(pixel_rf: RandomForestRegressor, size: int) -> np.ndarray:
    """Evaluate the RF on a size^3 RGB grid -> (size, size, size, 3) LUT in [0, 1].

    The grid is RGB; we convert each grid point to LAB, run the forest, and
    convert back, yielding a direct RGB->RGB mapping suitable for trilinear
    interpolation at apply time (and for .cube export).
    """
    if size < 2:
        raise ValueError(f"LUT size must be >= 2, got {size}")
    axis = np.linspace(0, 255, size, dtype=np.float32)
    R, G, B = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_rgb = np.stack([R, G, B], axis=-1).astype(np.uint8)  # (size, size, size, 3)

    lab = _rgb_to_lab(grid_rgb.reshape(size, size * size, 3))
    pred = pixel_rf.predict(lab.reshape(-1, 3).astype(np.float32))
    out_rgb = _lab_to_rgb(pred.astype(np.float32).reshape(size, size * size, 3))
    return out_rgb.reshape(size, size, size, 3).astype(np.float32) / 255.0


def apply_rgb_lut(rgb_u8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply an (N, N, N, 3) RGB LUT to an RGB image via trilinear interpolation."""
    from scipy.ndimage import map_coordinates

    size = lut.shape[0]
    scale = (size - 1) / 255.0
    coords = rgb_u8.astype(np.float32) * scale  # each channel now in [0, size-1]
    r = coords[..., 0].ravel()
    g = coords[..., 1].ravel()
    b = coords[..., 2].ravel()

    out = np.empty((r.size, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = map_coordinates(lut[..., c], [r, g, b], order=1, mode="nearest")
    out = np.clip(out, 0.0, 1.0) * 255.0
    return out.reshape(rgb_u8.shape).astype(np.uint8)


def fit_pixel_rf(
    src_lab_pixels: np.ndarray,
    tgt_lab_pixels: np.ndarray,
    *,
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    max_depth: int = DEFAULT_RF_MAX_DEPTH,
    max_samples: int = DEFAULT_RF_MAX_SAMPLES,
    seed: int = 0,
) -> RandomForestRegressor:
    """Train a multi-output Random Forest mapping LAB pixels src -> tgt.

    Subsamples to `max_samples` to keep profile size and training time bounded.
    """
    if len(src_lab_pixels) != len(tgt_lab_pixels):
        raise ValueError("Source and target pixel counts must match")
    if len(src_lab_pixels) == 0:
        raise ValueError("Cannot train a Random Forest on zero samples")

    rng = np.random.default_rng(seed)
    n = len(src_lab_pixels)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        src_lab_pixels = src_lab_pixels[idx]
        tgt_lab_pixels = tgt_lab_pixels[idx]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(src_lab_pixels.astype(np.float32), tgt_lab_pixels.astype(np.float32))
    return rf


def _apply_curves_lab(lab_pixels: np.ndarray, curves: TonalCurves) -> np.ndarray:
    out = np.empty_like(lab_pixels)
    out[..., 0] = curves.L_poly(lab_pixels[..., 0])
    out[..., 1] = curves.a_poly(lab_pixels[..., 1])
    out[..., 2] = curves.b_poly(lab_pixels[..., 2])
    return out


def apply_cluster_transform(
    rgb_u8: np.ndarray,
    transform: ClusterTransform,
    *,
    use_rf: bool = True,
) -> np.ndarray:
    """Apply a cluster's learned transform to an RGB image.

    When `use_rf=True`, prefers the baked RF LUT (fast trilinear interp),
    then the raw RF (per-pixel, slow — used for profiles trained before LUT
    baking existed), then falls back to Reinhard + tonal curves (Method A).
    """
    # getattr guards profiles pickled before rf_lut existed.
    rf_lut = getattr(transform, "rf_lut", None)
    if use_rf and rf_lut is not None:
        return apply_rgb_lut(rgb_u8, rf_lut)

    lab = _rgb_to_lab(rgb_u8)
    if use_rf and transform.pixel_rf is not None:
        h, w, _ = lab.shape
        pred = transform.pixel_rf.predict(lab.reshape(-1, 3).astype(np.float32))
        lab = pred.astype(np.float32).reshape(h, w, 3)
    else:
        lab = _apply_reinhard_lab(lab, transform.reinhard)
        lab = _apply_curves_lab(lab, transform.curves)
    return _lab_to_rgb(lab)
