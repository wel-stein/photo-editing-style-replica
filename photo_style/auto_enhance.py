"""Quick auto-enhance pipeline (independent of the trained profile flow).

Classical algorithms only — no deep learning. Six adjustments stack in
photographic order: white balance → exposure → shadows → highlights →
local contrast → saturation. Each takes a strength in [0, 1] where 0
disables that step and 1 applies the maximum effect.

`suggest_defaults` analyzes a photo's histogram and proposes reasonable
slider values (Lightroom-style "Auto" button).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    """All-in-one parameter bundle for auto_enhance(). Each in [0, 1]."""

    white_balance: float = 0.0
    exposure: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    local_contrast: float = 0.0
    saturation: float = 0.0


# ---------- Individual operations (each preserves shape + dtype) ----------


def auto_white_balance(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gray-world white balance: per-channel gain so the average pixel is gray.

    `strength` lerps between identity (0) and full gray-world correction (1).
    """
    if strength <= 0:
        return rgb_u8
    arr = rgb_u8.astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    gray_target = float(means.mean())
    gains = gray_target / (means + 1e-6)
    gains_lerped = 1.0 * (1 - strength) + gains * strength
    out = arr * gains_lerped[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_exposure(
    rgb_u8: np.ndarray, strength: float = 1.0, target_L: float = 128.0
) -> np.ndarray:
    """Brighten / darken so mean LAB-L approaches `target_L`.

    Works in both directions (darkens overexposed shots, brightens dark ones).
    """
    if strength <= 0:
        return rgb_u8
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    L_mean = float(L.mean())
    if L_mean < 1.0:
        return rgb_u8
    factor = target_L / L_mean
    factor_lerped = 1.0 * (1 - strength) + factor * strength
    lab[..., 0] = np.clip(L * factor_lerped, 0, 255)
    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)


def _shadow_lift_lut(strength: float) -> np.ndarray:
    """256-entry LUT that lifts shadows without touching midtones / highlights."""
    x = np.arange(256, dtype=np.float32) / 255.0
    gamma = 1.0 - 0.4 * strength  # gamma<1 lifts darks
    lifted = np.power(x, gamma)
    mask = np.clip(1.0 - x / 0.6, 0.0, 1.0)  # 1 at black, 0 by midtone
    y = x * (1.0 - mask * strength) + lifted * (mask * strength)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


def _highlight_recovery_lut(strength: float) -> np.ndarray:
    """256-entry LUT that compresses highlights without touching midtones / shadows."""
    x = np.arange(256, dtype=np.float32) / 255.0
    gamma = 1.0 + 0.5 * strength  # gamma>1 pulls highlights down
    compressed = np.power(x, gamma)
    mask = np.clip((x - 0.4) / 0.6, 0.0, 1.0)  # 0 by midtone, 1 at white
    y = x * (1.0 - mask * strength) + compressed * (mask * strength)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


def _apply_L_lut(rgb_u8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def lift_shadows(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Lift dark values; midtones and highlights unaffected."""
    if strength <= 0:
        return rgb_u8
    return _apply_L_lut(rgb_u8, _shadow_lift_lut(float(strength)))


def recover_highlights(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Compress bright values; midtones and shadows unaffected."""
    if strength <= 0:
        return rgb_u8
    return _apply_L_lut(rgb_u8, _highlight_recovery_lut(float(strength)))


def boost_saturation(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Increase HSV saturation. `strength=1` ~= +60%."""
    if strength <= 0:
        return rgb_u8
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + 0.6 * strength
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    hsv_u8 = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def apply_local_contrast(rgb_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """CLAHE on the L channel; clip limit scales with strength."""
    if strength <= 0:
        return rgb_u8
    clip_limit = 1.0 + 3.0 * float(strength)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(lab[..., 0])
    L_blended = lab[..., 0].astype(np.float32) * (1 - strength) + L_enhanced.astype(np.float32) * strength
    lab[..., 0] = np.clip(L_blended, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------- Composite pipeline ----------


def auto_enhance(rgb_u8: np.ndarray, params: EnhanceParams) -> np.ndarray:
    """Apply the six adjustments in photographic order. Each strength in [0, 1]."""
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb_u8.shape}")
    if rgb_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {rgb_u8.dtype}")
    out = rgb_u8
    out = auto_white_balance(out, params.white_balance)
    out = auto_exposure(out, params.exposure)
    out = lift_shadows(out, params.shadows)
    out = recover_highlights(out, params.highlights)
    out = apply_local_contrast(out, params.local_contrast)
    out = boost_saturation(out, params.saturation)
    return out


def suggest_defaults(rgb_u8: np.ndarray) -> EnhanceParams:
    """Histogram-driven 'Auto' button: analyzes the photo and proposes params."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)

    L_mean = float(L.mean())
    L_std = float(L.std())

    # Exposure: only correct if mean is far from neutral.
    if abs(L_mean - 128.0) > 15.0:
        exposure = min(1.0, abs(L_mean - 128.0) / 60.0)
    else:
        exposure = 0.0

    # Shadows: lift if a significant fraction of pixels are crushed near black.
    crushed = float((L < 20).sum()) / L.size
    shadows = min(1.0, crushed * 5.0)

    # Highlights: recover if many pixels are blown.
    clipped = float((L > 240).sum()) / L.size
    highlights = min(1.0, clipped * 5.0)

    # Local contrast: more aggressive when the L histogram is flat.
    local_contrast = max(0.0, min(0.8, (45.0 - L_std) / 45.0))

    # Modest defaults — these are safe on virtually any photo.
    saturation = 0.3
    white_balance = 0.4

    return EnhanceParams(
        white_balance=white_balance,
        exposure=exposure,
        shadows=shadows,
        highlights=highlights,
        local_contrast=local_contrast,
        saturation=saturation,
    )
