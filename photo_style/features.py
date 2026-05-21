"""Per-photo feature extraction for scene clustering.

Each photo is summarized into a fixed-length numeric vector that K-Means can
group by lighting/scene context. Features cover:

- brightness (LAB L mean & std)
- color cast (LAB a/b mean & std — proxies for tint and color temperature)
- tonal & color distribution (8-bin histograms of L, a, b)

Large images are downsampled before feature extraction so this stays cheap
even on 24 MP RAWs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NUM_HIST_BINS = 8
DEFAULT_MAX_DIM = 512


@dataclass
class PhotoFeatures:
    """Fixed-size summary of one photo's color/tone distribution in LAB."""

    L_mean: float
    L_std: float
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float
    L_hist: np.ndarray  # shape (NUM_HIST_BINS,), normalized to sum=1
    a_hist: np.ndarray
    b_hist: np.ndarray

    def to_vector(self) -> np.ndarray:
        """Flatten to a 1-D float32 vector suitable for K-Means."""
        return np.concatenate(
            [
                np.array(
                    [self.L_mean, self.L_std, self.a_mean, self.a_std, self.b_mean, self.b_std],
                    dtype=np.float32,
                ),
                self.L_hist.astype(np.float32),
                self.a_hist.astype(np.float32),
                self.b_hist.astype(np.float32),
            ]
        )

    @staticmethod
    def feature_names() -> list[str]:
        scalars = ["L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std"]
        hists = [f"{ch}_hist_{i}" for ch in ("L", "a", "b") for i in range(NUM_HIST_BINS)]
        return scalars + hists


def feature_vector_dim() -> int:
    """Return the length of the vector produced by PhotoFeatures.to_vector()."""
    return 6 + 3 * NUM_HIST_BINS


def _downsample(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return rgb
    scale = max_dim / longest
    return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _normalized_hist(channel: np.ndarray, num_bins: int = NUM_HIST_BINS) -> np.ndarray:
    counts, _ = np.histogram(channel, bins=num_bins, range=(0, 255))
    counts = counts.astype(np.float32)
    total = counts.sum()
    return counts / total if total > 0 else counts


def extract_features(rgb_u8: np.ndarray, max_dim: int = DEFAULT_MAX_DIM) -> PhotoFeatures:
    """Compute LAB summary features for one HxWx3 uint8 RGB image."""
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb_u8.shape}")
    if rgb_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {rgb_u8.dtype}")

    small = _downsample(rgb_u8, max_dim)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    return PhotoFeatures(
        L_mean=float(L.mean()),
        L_std=float(L.std()),
        a_mean=float(a.mean()),
        a_std=float(a.std()),
        b_mean=float(b.mean()),
        b_std=float(b.std()),
        L_hist=_normalized_hist(L),
        a_hist=_normalized_hist(a),
        b_hist=_normalized_hist(b),
    )
