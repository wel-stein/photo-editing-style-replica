"""Single-photo inference: feature extraction -> cluster assignment -> apply."""

from __future__ import annotations

import numpy as np

from .features import extract_features
from .profile_store import StyleProfile
from .style_model import apply_cluster_transform


def apply_profile(
    rgb_u8: np.ndarray,
    profile: StyleProfile,
    *,
    use_rf: bool = True,
    override_cluster: int | None = None,
) -> tuple[np.ndarray, int]:
    """Style an image with a saved profile.

    Returns (styled_rgb, cluster_id_used). When `override_cluster` is given,
    skips feature-based assignment and uses that cluster directly (useful for
    LUT export and previewing each cluster's look).
    """
    if override_cluster is None:
        feature_vec = extract_features(rgb_u8).to_vector()
        cluster_id = int(profile.cluster_model.assign(feature_vec)[0])
    else:
        cluster_id = int(override_cluster)

    if cluster_id not in profile.cluster_transforms:
        available = sorted(profile.cluster_transforms.keys())
        raise KeyError(
            f"Cluster {cluster_id} not in profile (available: {available})"
        )
    transform = profile.cluster_transforms[cluster_id]
    styled = apply_cluster_transform(rgb_u8, transform, use_rf=use_rf)
    return styled, cluster_id
