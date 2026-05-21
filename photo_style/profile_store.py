"""Save/load named style profiles as `.pkl` + `.json` manifest.

Profile contents (the `.pkl`):
- the fitted ClusterModel (scaler + KMeans) so new photos can be assigned
- one ClusterTransform per cluster (Reinhard stats + tonal curves)
- training metadata mirror of the manifest

The manifest (`.json`) is human-readable: training date, photo count, cluster
count, and a placeholder for Phase 4 validation metrics (ΔE).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .clustering import ClusterModel
from .style_model import ClusterTransform

DEFAULT_PROFILES_DIR = Path("profiles")


@dataclass
class ProfileMetadata:
    name: str
    created_at: str
    photo_count: int
    cluster_count: int
    samples_per_pair: int
    feature_max_dim: int
    schema_version: int = 1
    validation_delta_e: float | None = None  # filled in Phase 4
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleProfile:
    metadata: ProfileMetadata
    cluster_model: ClusterModel
    cluster_transforms: dict[int, ClusterTransform]


def _profile_paths(name: str, profiles_dir: Path) -> tuple[Path, Path]:
    safe = name.replace("/", "_").replace("\\", "_")
    return profiles_dir / f"{safe}.pkl", profiles_dir / f"{safe}.json"


def save_profile(profile: StyleProfile, profiles_dir: str | Path = DEFAULT_PROFILES_DIR) -> Path:
    """Write the profile pickle and manifest. Returns the pickle path."""
    profiles_dir = Path(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    pkl_path, json_path = _profile_paths(profile.metadata.name, profiles_dir)

    joblib.dump(
        {
            "cluster_model": profile.cluster_model,
            "cluster_transforms": profile.cluster_transforms,
            "metadata": asdict(profile.metadata),
        },
        pkl_path,
        compress=3,
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(profile.metadata), f, indent=2)
    return pkl_path


def load_profile(name: str, profiles_dir: str | Path = DEFAULT_PROFILES_DIR) -> StyleProfile:
    """Load a previously saved profile by name."""
    profiles_dir = Path(profiles_dir)
    pkl_path, _ = _profile_paths(name, profiles_dir)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No profile found at {pkl_path}")
    obj = joblib.load(pkl_path)
    metadata = ProfileMetadata(**obj["metadata"])
    return StyleProfile(
        metadata=metadata,
        cluster_model=obj["cluster_model"],
        cluster_transforms=obj["cluster_transforms"],
    )


def list_profiles(profiles_dir: str | Path = DEFAULT_PROFILES_DIR) -> list[str]:
    """List names of saved profiles (by .pkl files in the directory)."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.pkl"))


def make_metadata(
    name: str,
    photo_count: int,
    cluster_count: int,
    samples_per_pair: int,
    feature_max_dim: int,
) -> ProfileMetadata:
    return ProfileMetadata(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        photo_count=photo_count,
        cluster_count=cluster_count,
        samples_per_pair=samples_per_pair,
        feature_max_dim=feature_max_dim,
    )
