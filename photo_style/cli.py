"""Command-line interface for training, applying, and inspecting style profiles.

Subcommands:
- train:  train a profile from a folder of pairs (Phase 3 + RF + validation)
- apply:  style a single image with a saved profile (Phase 4)
- list:   list saved profiles
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from .clustering import fit_clusters
from .evaluation import evaluate_profile_on_pairs
from .feature_cache import FeatureCache
from .features import DEFAULT_MAX_DIM
from .inference import apply_profile
from .io_utils import find_pairs, load_image_rgb, save_image_rgb
from .lut_export import DEFAULT_LUT_SIZE, export_profile_lut
from .profile_store import (
    DEFAULT_PROFILES_DIR,
    StyleProfile,
    list_profiles,
    load_profile,
    make_metadata,
    save_profile,
)
from .style_model import (
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MAX_SAMPLES,
    DEFAULT_RF_N_ESTIMATORS,
    fit_cluster_transform,
    sample_pair_pixels,
)

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

    # Train/val split (deterministic given --seed)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(pair_result.num_pairs)
    rng.shuffle(indices)
    val_count = int(round(args.val_fraction * pair_result.num_pairs))
    val_count = max(0, min(val_count, pair_result.num_pairs - 1))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    train_pairs = [pair_result.pairs[i] for i in train_idx]
    val_pairs = [pair_result.pairs[i] for i in val_idx]
    logger.info("Split: {} train / {} val", len(train_pairs), len(val_pairs))

    cache = FeatureCache(args.cache_path)
    feature_vectors: list[np.ndarray] = []
    logger.info("Extracting features for {} training originals (cache: {})", len(train_pairs), args.cache_path)
    for i, (src, _) in enumerate(train_pairs, start=1):
        vec = cache.get_or_compute(src, max_dim=args.feature_max_dim)
        feature_vectors.append(vec)
        if i % 25 == 0 or i == len(train_pairs):
            logger.info("  features {}/{}", i, len(train_pairs))
    cache.save()

    features = np.stack(feature_vectors).astype(np.float32)
    cluster_model = fit_clusters(features, k=args.k, random_state=args.seed)
    logger.info(
        "Clustered {} photos into {} groups (sizes: {})",
        features.shape[0],
        cluster_model.n_clusters,
        np.bincount(cluster_model.labels, minlength=cluster_model.n_clusters).tolist(),
    )

    fit_rf = not args.no_rf
    rng_pixels = np.random.default_rng(args.seed)
    cluster_transforms = {}
    for c in range(cluster_model.n_clusters):
        member_idx = np.where(cluster_model.labels == c)[0]
        if len(member_idx) == 0:
            logger.warning("Cluster {} is empty; skipping", c)
            continue
        logger.info(
            "Cluster {}: sampling pixels from {} pair(s){}",
            c,
            len(member_idx),
            " + training RF" if fit_rf else "",
        )
        pair_pixels = []
        for i in member_idx:
            src_path, tgt_path = train_pairs[i]
            src_rgb = load_image_rgb(src_path)
            tgt_rgb = load_image_rgb(tgt_path)
            s_pix, t_pix = sample_pair_pixels(
                src_rgb, tgt_rgb, n_samples=args.samples_per_pair, rng=rng_pixels
            )
            pair_pixels.append((s_pix, t_pix))
        cluster_transforms[c] = fit_cluster_transform(
            pair_pixels,
            fit_rf=fit_rf,
            rf_n_estimators=args.rf_trees,
            rf_max_depth=args.rf_max_depth,
            rf_max_samples=args.rf_max_samples,
            seed=args.seed + c,
        )

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

    if val_pairs:
        logger.info("Evaluating on {} held-out pair(s)...", len(val_pairs))
        metrics = evaluate_profile_on_pairs(profile, val_pairs)
        method_b_str = f"{metrics.method_b:.2f}" if metrics.method_b is not None else "n/a"
        logger.info(
            "Mean ΔE2000 over {} val pairs: identity={:.2f}  methodA={:.2f}  methodB={}",
            metrics.n_pairs,
            metrics.identity,
            metrics.method_a,
            method_b_str,
        )
        profile.metadata.validation_delta_e = (
            metrics.method_b if metrics.method_b is not None else metrics.method_a
        )
        profile.metadata.extra["validation_metrics"] = metrics.as_dict()
    else:
        logger.info("No validation set (val_fraction={}) -- skipping ΔE metrics", args.val_fraction)

    pkl_path = save_profile(profile, profiles_dir=args.profiles_dir)
    size_mb = pkl_path.stat().st_size / (1024 * 1024)
    elapsed = time.perf_counter() - t0
    logger.success(
        "Saved profile to {} ({:.1f} MB, training took {:.1f}s)", pkl_path, size_mb, elapsed
    )
    if size_mb > 50:
        logger.warning(
            "Profile exceeds the 50 MB target. Consider lowering --rf-trees, --rf-max-depth, "
            "or --rf-max-samples."
        )
    return 0


def _apply(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile, profiles_dir=args.profiles_dir)
    rgb = load_image_rgb(args.input)
    logger.info("Loaded image {} (shape={})", args.input, rgb.shape)
    use_rf = not args.no_rf
    t0 = time.perf_counter()
    styled, cluster_id = apply_profile(
        rgb, profile, use_rf=use_rf, override_cluster=args.cluster
    )
    elapsed = time.perf_counter() - t0
    save_image_rgb(styled, args.output, quality=args.quality, source_path=args.input)
    logger.success(
        "Applied profile '{}' (cluster {}, method {}) in {:.1f}s -> {}",
        args.profile,
        cluster_id,
        "B (RF)" if (use_rf and profile.cluster_transforms[cluster_id].pixel_rf is not None) else "A",
        elapsed,
        args.output,
    )
    return 0


def _export_lut(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile, profiles_dir=args.profiles_dir)
    cluster: int | str
    cluster = "average" if args.cluster == "average" else int(args.cluster)
    t0 = time.perf_counter()
    out_path = export_profile_lut(
        profile,
        args.output,
        cluster=cluster,
        size=args.size,
        use_rf=not args.no_rf,
    )
    elapsed = time.perf_counter() - t0
    logger.success(
        "Wrote {}x{}x{} LUT to {} (cluster={}, method={}) in {:.1f}s",
        args.size,
        args.size,
        args.size,
        out_path,
        cluster,
        "B (RF)" if not args.no_rf else "A",
        elapsed,
    )
    print()
    print("Install in Lightroom Classic:")
    print("  Lightroom cannot read .cube directly; convert to .xmp Camera Raw profile first.")
    print("  Easiest path: use Photoshop > Camera Raw Filter > Profile dropdown > 'Browse' > select wrapper .xmp,")
    print("  OR for a quick visual test of the LUT itself:")
    print("    Photoshop > File > Open your photo > Image > Adjustments > Color Lookup")
    print("    > Load 3D LUT > select this .cube. Verify the look matches before wrapping for Lightroom.")
    print()
    print("  Wrap .cube as .xmp for Lightroom Classic:")
    print("    - free option: https://github.com/cawtekfilms/lutbaker  (cube -> xmp)")
    print("    - then drop the .xmp into:")
    print("        Win: %APPDATA%\\Adobe\\CameraRaw\\Settings\\")
    print("        Mac: ~/Library/Application Support/Adobe/CameraRaw/Settings/")
    print("    - restart Lightroom; the profile appears under Develop > Profile > Browse > User Profiles.")
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
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of pairs held out for ΔE2000 validation (default 0.2)",
    )
    train.add_argument("--no-rf", action="store_true", help="Skip the Random Forest (Method B)")
    train.add_argument(
        "--rf-trees", type=int, default=DEFAULT_RF_N_ESTIMATORS, help="RF n_estimators"
    )
    train.add_argument(
        "--rf-max-depth", type=int, default=DEFAULT_RF_MAX_DEPTH, help="RF max_depth"
    )
    train.add_argument(
        "--rf-max-samples",
        type=int,
        default=DEFAULT_RF_MAX_SAMPLES,
        help="Maximum pooled pixel samples per cluster fed to the RF",
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

    apply_p = sub.add_parser("apply", help="Apply a saved profile to a single image.")
    apply_p.add_argument("--profile", required=True, help="Profile name (without extension)")
    apply_p.add_argument("--input", required=True, type=Path, help="Input image (JPG/RAW/PNG)")
    apply_p.add_argument("--output", required=True, type=Path, help="Output JPEG path")
    apply_p.add_argument(
        "--cluster", type=int, default=None, help="Force a specific cluster (skip feature-based assignment)"
    )
    apply_p.add_argument("--no-rf", action="store_true", help="Use Method A (Reinhard + curves) only")
    apply_p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    apply_p.add_argument(
        "--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR, help="Where profiles live"
    )
    apply_p.set_defaults(func=_apply)

    export = sub.add_parser("export-lut", help="Export a profile as a .cube 3D LUT.")
    export.add_argument("--profile", required=True, help="Profile name (without extension)")
    export.add_argument("--output", required=True, type=Path, help="Path to write the .cube file")
    export.add_argument(
        "--cluster",
        default="average",
        help="Cluster index (e.g. 0) or 'average' for a single global LUT",
    )
    export.add_argument(
        "--size", type=int, default=DEFAULT_LUT_SIZE, help="LUT side length (default 33)"
    )
    export.add_argument("--no-rf", action="store_true", help="Bake Method A instead of RF")
    export.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    export.set_defaults(func=_export_lut)

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
