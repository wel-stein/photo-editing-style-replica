"""Export learned style transforms as 33x33x33 .cube LUTs.

A 3D LUT samples the input sRGB cube on a regular grid and stores the styled
output for each grid point. Tools that consume LUTs (Lightroom via a wrapper,
Photoshop, DaVinci Resolve) trilinearly interpolate between grid points at
apply time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .profile_store import StyleProfile
from .style_model import ClusterTransform, apply_cluster_transform

DEFAULT_LUT_SIZE = 33


def generate_lut_array(
    transform: ClusterTransform,
    *,
    size: int = DEFAULT_LUT_SIZE,
    use_rf: bool = True,
) -> np.ndarray:
    """Build the (size, size, size, 3) LUT in [0, 1] for one cluster transform.

    Indexing: lut[r_idx, g_idx, b_idx] -> output RGB.
    """
    if size < 2:
        raise ValueError(f"LUT size must be >= 2, got {size}")
    axis = np.linspace(0, 255, size, dtype=np.float32)
    R, G, B = np.meshgrid(axis, axis, axis, indexing="ij")
    input_grid = np.stack([R, G, B], axis=-1).astype(np.uint8)  # (size, size, size, 3)

    # Reshape into a 2D "image" so apply_cluster_transform can run unchanged.
    img = input_grid.reshape(size, size * size, 3)
    styled = apply_cluster_transform(img, transform, use_rf=use_rf)
    return styled.reshape(size, size, size, 3).astype(np.float32) / 255.0


def generate_average_lut_array(
    profile: StyleProfile,
    *,
    size: int = DEFAULT_LUT_SIZE,
    use_rf: bool = True,
) -> np.ndarray:
    """Average per-cluster LUTs into a single global LUT.

    Caveat: averaging the *output* of N nonlinear transforms is not the same
    as a single transform learned globally; it's a reasonable approximation
    when the user wants one LUT instead of N.
    """
    if not profile.cluster_transforms:
        raise ValueError("Profile has no cluster transforms")
    luts = [
        generate_lut_array(t, size=size, use_rf=use_rf)
        for t in profile.cluster_transforms.values()
    ]
    return np.mean(np.stack(luts, axis=0), axis=0)


def write_cube(
    lut: np.ndarray,
    path: str | Path,
    *,
    title: str = "photo_style_ai",
) -> Path:
    """Write a 3D LUT to .cube format.

    `lut` shape must be (N, N, N, 3) with values in [0, 1]; indexed [r, g, b].
    `.cube` byte order: R varies fastest, then G, then B varies slowest.
    """
    if lut.ndim != 4 or lut.shape[0] != lut.shape[1] != lut.shape[2] or lut.shape[3] != 3:
        raise ValueError(f"LUT must be (N, N, N, 3); got {lut.shape}")
    size = int(lut.shape[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reorder so that the flat traversal is [b][g][r] (b outermost, r innermost).
    reordered = lut.transpose(2, 1, 0, 3)
    flat = np.clip(reordered.reshape(-1, 3), 0.0, 1.0)

    with path.open("w", encoding="ascii") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(f, flat, fmt="%.6f", delimiter=" ")
    return path


def export_profile_lut(
    profile: StyleProfile,
    output_path: str | Path,
    *,
    cluster: int | str = "average",
    size: int = DEFAULT_LUT_SIZE,
    use_rf: bool = True,
) -> Path:
    """High-level helper used by the CLI: pick a cluster (or 'average') and write."""
    if cluster == "average":
        lut = generate_average_lut_array(profile, size=size, use_rf=use_rf)
        title = f"{profile.metadata.name}_average"
    else:
        cluster_id = int(cluster)
        if cluster_id not in profile.cluster_transforms:
            available = sorted(profile.cluster_transforms.keys())
            raise KeyError(f"Cluster {cluster_id} not in profile (available: {available})")
        lut = generate_lut_array(
            profile.cluster_transforms[cluster_id], size=size, use_rf=use_rf
        )
        title = f"{profile.metadata.name}_cluster{cluster_id}"
    return write_cube(lut, output_path, title=title)
