"""Command-line interface for training and inspecting style profiles.

Phase 3 covers the `train` subcommand. Inference (`apply`) and LUT export
(`export-lut`) land in Phases 4 and 5.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from .clustering import fit_clusters
from .feature_cache import FeatureCache
from .features import DEFAULT_MAX_DIM
from .io_utils import find_pairs, load_image_rgb
from .profile_store import (
    DEFAULT_PROFILES_DIR,
    StyleProfile,
    list_profiles,
    make_metadata,
    save_profile,
)
from .style_model import fit_cluster_transform, sample_pair_pixels

DEFAULT_CACHE_PATH = Path(".cache/features.pkl")


def _train(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    pair_result = find_pairs(args.originals, args.edited)
    logger.info(
        "Pair discovery: {} matched, {} unmatched originals, {} unmatched edits",
        pair_result.num_pairs,
        len(pair_result.unmatched_originals),
        len(pair_result.unmatched_edited),
    )
    if pair_result.num_pairs == 0:
        logger.error("No matched pairs found. Check filename stems across the two folders.")
        return 1
    for p in pair_result.unmatched_originals:
        logger.warning("Unmatched original: {}", p.name)
    for p in pair_result.unmatched_edited:
        logger.warning("Unmatched edit: {}", p.name)

    cache = FeatureCache(args.cache_path)
    feature_vectors: list[np.ndarray] = []
    logger.info("Extracting features for {} originals (cache: {})", pair_result.num_pairs, args.cache_path)
    for i, (src, _) in enumerate(pair_result.pairs, start=1):
        vec = cache.get_or_compute(src, max_dim=args.feature_max_dim)
        feature_vectors.append(vec)
        if i % 25 == 0 or i == pair_result.num_pairs:
            logger.info("  features {}/{}", i, pair_result.num_pairs)
    cache.save()

    features = np.stack(feature_vectors).astype(np.float32)
    cluster_model = fit_clusters(features, k=args.k, random_state=args.seed)
    logger.info(
        "Clustered {} photos into {} groups (sizes: {})",
        features.shape[0],
        cluster_model.n_clusters,
        np.bincount(cluster_model.labels, minlength=cluster_model.n_clusters).tolist(),
    )

    rng = np.random.default_rng(args.seed)
    cluster_transforms = {}
    for c in range(cluster_model.n_clusters):
        member_idx = np.where(cluster_model.labels == c)[0]
        if len(member_idx) == 0:
            logger.warning("Cluster {} is empty; skipping", c)
            continue
        logger.info("Cluster {}: fitting transform from {} pair(s)", c, len(member_idx))
        pair_pixels = []
        for i in member_idx:
            src_path, tgt_path = pair_result.pairs[i]
            src_rgb = load_image_rgb(src_path)
            tgt_rgb = load_image_rgb(tgt_path)
            s_pix, t_pix = sample_pair_pixels(
                src_rgb, tgt_rgb, n_samples=args.samples_per_pair, rng=rng
            )
            pair_pixels.append((s_pix, t_pix))
        cluster_transforms[c] = fit_cluster_transform(pair_pixels)

    metadata = make_metadata(
        name=args.name,
        photo_count=pair_result.num_pairs,
        cluster_count=cluster_model.n_clusters,
        samples_per_pair=args.samples_per_pair,
        feature_max_dim=args.feature_max_dim,
    )
    profile = StyleProfile(
        metadata=metadata,
        cluster_model=cluster_model,
        cluster_transforms=cluster_transforms,
    )
    pkl_path = save_profile(profile, profiles_dir=args.profiles_dir)
    elapsed = time.perf_counter() - t0
    logger.success("Saved profile to {} (training took {:.1f}s)", pkl_path, elapsed)
    return 0


def _list(args: argparse.Namespace) -> int:
    names = list_profiles(args.profiles_dir)
    if not names:
        logger.info("No profiles found in {}", args.profiles_dir)
        return 0
    for n in names:
        print(n)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-style-ai",
        description="Train and manage personal photo style profiles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a style profile from a folder of pairs.")
    train.add_argument("--originals", required=True, type=Path, help="Folder of original photos")
    train.add_argument("--edited", required=True, type=Path, help="Folder of edited photos")
    train.add_argument("--name", required=True, help="Profile name (used for the .pkl filename)")
    train.add_argument("--k", type=int, default=4, help="Number of clusters (3-7, default 4)")
    train.add_argument(
        "--samples-per-pair",
        type=int,
        default=10000,
        help="Pixels sampled per pair when fitting transforms (5k-50k)",
    )
    train.add_argument(
        "--feature-max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Longest-edge resize for feature extraction",
    )
    train.add_argument(
        "--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR, help="Where to write the profile"
    )
    train.add_argument(
        "--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="Feature cache pickle path"
    )
    train.add_argument("--seed", type=int, default=0, help="Random seed for K-Means + sampling")
    train.set_defaults(func=_train)

    ls = sub.add_parser("list", help="List saved profiles.")
    ls.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    ls.set_defaults(func=_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
