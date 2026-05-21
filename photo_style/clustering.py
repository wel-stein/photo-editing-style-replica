"""K-Means clustering over per-photo feature vectors.

Groups training pairs by lighting/scene context so each cluster can learn its
own color transform. Feature vectors are standardized first (zero mean, unit
variance per dimension) so histogram bins don't drown out scalar moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterModel:
    """Fitted clustering pipeline: scaler + KMeans. Re-used at inference."""

    scaler: StandardScaler
    kmeans: KMeans
    labels: np.ndarray  # cluster assignment for each training photo

    @property
    def n_clusters(self) -> int:
        return int(self.kmeans.n_clusters)

    def assign(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Assign one or more new feature vectors to their nearest cluster."""
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)
        return self.kmeans.predict(self.scaler.transform(feature_vectors))


def fit_clusters(feature_vectors: np.ndarray, k: int, random_state: int = 0) -> ClusterModel:
    """Standardize features then run K-Means with the given cluster count.

    `k` is clamped to [1, len(feature_vectors)] so this also works on tiny
    training sets (useful for tests and early experiments).
    """
    if feature_vectors.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape {feature_vectors.shape}")
    n = feature_vectors.shape[0]
    if n == 0:
        raise ValueError("Cannot cluster an empty feature matrix")
    effective_k = max(1, min(k, n))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_vectors)

    kmeans = KMeans(n_clusters=effective_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled)

    return ClusterModel(scaler=scaler, kmeans=kmeans, labels=labels)
