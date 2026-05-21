"""Persistent cache for per-photo feature vectors.

Keyed on (absolute path, mtime, file size, max_dim). Re-extracting features
from a 24 MP RAW is the slowest step in training, so this lets retraining
with different cluster counts feel instant.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .features import DEFAULT_MAX_DIM, extract_features
from .io_utils import load_image_rgb

CacheKey = tuple[str, float, int, int]


class FeatureCache:
    """Tiny pickle-backed dict mapping (path, mtime, size, max_dim) -> vector."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._data: dict[CacheKey, np.ndarray] = self._load()
        self._dirty = False

    def _load(self) -> dict[CacheKey, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {}
        except (pickle.PickleError, EOFError, OSError):
            return {}

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self._data, f)
        self._dirty = False

    def get_or_compute(self, path: Path, max_dim: int = DEFAULT_MAX_DIM) -> np.ndarray:
        path = Path(path)
        stat = path.stat()
        key: CacheKey = (str(path.resolve()), stat.st_mtime, stat.st_size, max_dim)
        if key in self._data:
            return self._data[key]
        vec = extract_features(load_image_rgb(path), max_dim=max_dim).to_vector()
        self._data[key] = vec
        self._dirty = True
        return vec
