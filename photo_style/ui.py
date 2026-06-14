"""Gradio UI with three tabs: Train, Apply, Export LUT.

Data-logic helpers (the `_helper_*` functions) are pure: they accept primitive
inputs, do the work, and return primitive outputs. The `build_ui` function
wires them to Gradio components and adds progress callbacks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import numpy as np
from loguru import logger

from .auto_enhance import EnhanceParams, auto_enhance, suggest_defaults
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
    fit_cluster_transform,
    sample_pair_pixels,
)

CACHE_PATH = Path(".cache/features.pkl")


# ---------- Pure helpers (testable without Gradio) ----------


def helper_match_pairs(originals_dir: str, edited_dir: str) -> tuple[str, list[str]]:
    """Return (status message, list of unmatched file names) for the UI."""
    if not originals_dir or not edited_dir:
        return "Please enter both folder paths.", []
    result = find_pairs(originals_dir, edited_dir)
    msg = (
        f"Matched {result.num_pairs} pair(s). "
        f"{len(result.unmatched_originals)} unmatched original(s), "
        f"{len(result.unmatched_edited)} unmatched edit(s)."
    )
    unmatched = [f"original: {p.name}" for p in result.unmatched_originals]
    unmatched += [f"edit: {p.name}" for p in result.unmatched_edited]
    return msg, unmatched


def helper_train(
    originals_dir: str,
    edited_dir: str,
    k: int,
    samples_per_pair: int,
    val_fraction: float,
    use_rf: bool,
    profile_name: str,
    *,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    cache_path: Path = CACHE_PATH,
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """Train a profile end-to-end.

    Returns (status_text, features_for_plot, labels_for_plot). Caller renders
    the plot since matplotlib import is expensive and Gradio-specific.
    """
    if not profile_name.strip():
        return "Please enter a profile name.", None, None

    def _p(frac: float, desc: str) -> None:
        if progress is not None:
            progress(frac, desc)

    _p(0.02, "Discovering pairs")
    pair_result = find_pairs(originals_dir, edited_dir)
    if pair_result.num_pairs == 0:
        return "No matched pairs found. Check filename stems across the two folders.", None, None

    rng = np.random.default_rng(seed)
    indices = np.arange(pair_result.num_pairs)
    rng.shuffle(indices)
    val_count = max(0, min(int(round(val_fraction * pair_result.num_pairs)), pair_result.num_pairs - 1))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    train_pairs = [pair_result.pairs[i] for i in train_idx]
    val_pairs = [pair_result.pairs[i] for i in val_idx]

    _p(0.05, f"Extracting features ({len(train_pairs)} pairs)")
    cache = FeatureCache(cache_path)
    feature_vectors: list[np.ndarray] = []
    for i, (src, _) in enumerate(train_pairs, start=1):
        feature_vectors.append(cache.get_or_compute(src, max_dim=DEFAULT_MAX_DIM))
        _p(0.05 + 0.35 * i / len(train_pairs), f"Features {i}/{len(train_pairs)}")
    cache.save()

    _p(0.45, "Clustering")
    features = np.stack(feature_vectors).astype(np.float32)
    cluster_model = fit_clusters(features, k=int(k), random_state=seed)

    rng_pixels = np.random.default_rng(seed)
    cluster_transforms: dict[int, Any] = {}
    n_clusters = cluster_model.n_clusters
    for c in range(n_clusters):
        member_idx = np.where(cluster_model.labels == c)[0]
        if len(member_idx) == 0:
            continue
        _p(
            0.5 + 0.4 * c / max(n_clusters, 1),
            f"Cluster {c + 1}/{n_clusters}: {len(member_idx)} pair(s)" + (" + RF" if use_rf else ""),
        )
        pair_pixels = []
        for i in member_idx:
            src_path, tgt_path = train_pairs[i]
            src_rgb = load_image_rgb(src_path)
            tgt_rgb = load_image_rgb(tgt_path)
            s_pix, t_pix = sample_pair_pixels(
                src_rgb, tgt_rgb, n_samples=int(samples_per_pair), rng=rng_pixels
            )
            pair_pixels.append((s_pix, t_pix))
        cluster_transforms[c] = fit_cluster_transform(
            pair_pixels, fit_rf=bool(use_rf), seed=seed + c
        )

    metadata = make_metadata(
        name=profile_name.strip(),
        photo_count=pair_result.num_pairs,
        cluster_count=cluster_model.n_clusters,
        samples_per_pair=int(samples_per_pair),
        feature_max_dim=DEFAULT_MAX_DIM,
    )
    profile = StyleProfile(
        metadata=metadata, cluster_model=cluster_model, cluster_transforms=cluster_transforms
    )

    de_line = "(no validation set)"
    if val_pairs:
        _p(0.92, f"Validating on {len(val_pairs)} pair(s)")
        metrics = evaluate_profile_on_pairs(profile, val_pairs)
        profile.metadata.validation_delta_e = (
            metrics.method_b if metrics.method_b is not None else metrics.method_a
        )
        profile.metadata.extra["validation_metrics"] = metrics.as_dict()
        method_b_str = f"{metrics.method_b:.2f}" if metrics.method_b is not None else "n/a"
        de_line = (
            f"ΔE2000 over {metrics.n_pairs} val pairs: "
            f"identity={metrics.identity:.2f}  methodA={metrics.method_a:.2f}  methodB={method_b_str}"
        )

    _p(0.97, "Saving profile")
    pkl_path = save_profile(profile, profiles_dir=profiles_dir)
    size_mb = pkl_path.stat().st_size / (1024 * 1024)
    _p(1.0, "Done")

    status = (
        f"Saved '{profile_name}' to {pkl_path} ({size_mb:.1f} MB).\n"
        f"{pair_result.num_pairs} pairs ({len(train_pairs)} train / {len(val_pairs)} val)\n"
        f"{cluster_model.n_clusters} clusters, sizes: "
        f"{np.bincount(cluster_model.labels, minlength=cluster_model.n_clusters).tolist()}\n"
        f"{de_line}"
    )
    return status, features, cluster_model.labels


def _cluster_plot(features: np.ndarray, labels: np.ndarray):
    """Project features to 2D via PCA and scatter by cluster. Returns a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    xy = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cluster assignments (PCA of features)")
    n_clusters = int(labels.max()) + 1
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"Cluster {i}" for i in range(n_clusters)], loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def helper_apply(
    profile_name: str,
    image_path: str | None,
    use_rf: bool,
    *,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str]:
    """Style an image with a saved profile.

    Returns (before_rgb, after_rgb, downloadable_path, status_message).
    """
    if not profile_name:
        return None, None, None, "Pick a profile first."
    if not image_path:
        return None, None, None, "Upload an image first."

    profile = load_profile(profile_name, profiles_dir=profiles_dir)
    rgb = load_image_rgb(image_path)
    styled, cluster_id = apply_profile(rgb, profile, use_rf=use_rf)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jpg", prefix=f"{profile_name}_styled_", delete=False
    )
    tmp.close()
    save_image_rgb(styled, tmp.name)

    method = "B (RF)" if (use_rf and profile.cluster_transforms[cluster_id].pixel_rf is not None) else "A"
    status = f"Applied profile '{profile_name}' using cluster {cluster_id}, method {method}."
    return rgb, styled, tmp.name, status


def helper_export_lut(
    profile_name: str,
    cluster_choice: str,
    size: int,
    use_rf: bool,
    *,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> tuple[str | None, str]:
    """Export a .cube file. Returns (downloadable_path, status_message)."""
    if not profile_name:
        return None, "Pick a profile first."
    profile = load_profile(profile_name, profiles_dir=profiles_dir)

    cluster: int | str
    if cluster_choice == "average":
        cluster = "average"
        suffix = "average"
    else:
        cluster = int(cluster_choice)
        suffix = f"cluster{cluster}"

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{suffix}.cube", prefix=f"{profile_name}_", delete=False
    )
    tmp.close()
    out_path = export_profile_lut(
        profile, tmp.name, cluster=cluster, size=int(size), use_rf=bool(use_rf)
    )
    status = f"Wrote {size}x{size}x{size} LUT to {out_path}."
    return str(out_path), status


def helper_auto_analyze(image_path: str | None) -> tuple[float, float, float, float, float, float, str]:
    """Analyze an uploaded image and propose enhancement slider values.

    Returns (wb, exposure, shadows, highlights, local_contrast, saturation, status).
    """
    if not image_path:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Upload a photo first."
    rgb = load_image_rgb(image_path)
    params = suggest_defaults(rgb)
    status = (
        f"Suggested: WB={params.white_balance:.2f}  Exp={params.exposure:.2f}  "
        f"Sh={params.shadows:.2f}  Hi={params.highlights:.2f}  "
        f"LC={params.local_contrast:.2f}  Sat={params.saturation:.2f}"
    )
    return (
        params.white_balance,
        params.exposure,
        params.shadows,
        params.highlights,
        params.local_contrast,
        params.saturation,
        status,
    )


def helper_auto_enhance(
    image_path: str | None,
    white_balance: float,
    exposure: float,
    shadows: float,
    highlights: float,
    local_contrast: float,
    saturation: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str]:
    """Apply the auto-enhance pipeline with the given slider values.

    Returns (before_rgb, after_rgb, downloadable_path, status_message).
    """
    if not image_path:
        return None, None, None, "Upload a photo first."
    rgb = load_image_rgb(image_path)
    params = EnhanceParams(
        white_balance=float(white_balance),
        exposure=float(exposure),
        shadows=float(shadows),
        highlights=float(highlights),
        local_contrast=float(local_contrast),
        saturation=float(saturation),
    )
    enhanced = auto_enhance(rgb, params)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="auto_enhanced_", delete=False)
    tmp.close()
    save_image_rgb(enhanced, tmp.name)

    status = (
        f"Applied: WB={params.white_balance:.2f}  Exp={params.exposure:.2f}  "
        f"Sh={params.shadows:.2f}  Hi={params.highlights:.2f}  "
        f"LC={params.local_contrast:.2f}  Sat={params.saturation:.2f}"
    )
    return rgb, enhanced, tmp.name, status


def _cluster_choices_for(profile_name: str, profiles_dir: Path = DEFAULT_PROFILES_DIR) -> list[str]:
    """Return ['average', '0', '1', ...] for the selected profile."""
    if not profile_name:
        return ["average"]
    try:
        profile = load_profile(profile_name, profiles_dir=profiles_dir)
    except FileNotFoundError:
        return ["average"]
    return ["average"] + [str(c) for c in sorted(profile.cluster_transforms.keys())]


# ---------- Gradio Blocks ----------


def build_ui(profiles_dir: Path = DEFAULT_PROFILES_DIR) -> gr.Blocks:
    with gr.Blocks(title="Photo Style AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Photo Style AI\n"
            "Learn a personal photo editing style from before/after pairs and apply it."
        )

        # Components defined first so events can reference any of them.
        with gr.Tabs():
            # ---------- Train ----------
            with gr.Tab("Train"):
                gr.Markdown(
                    "Place originals and your edited versions in two folders with **matching "
                    "filename stems** (e.g. `DSC_8701.NEF` ↔ `DSC_8701.jpg`)."
                )
                with gr.Row():
                    originals_dir = gr.Textbox(
                        label="Originals folder",
                        placeholder=r"C:\path\to\NEFs",
                        info=(
                            "Folder containing your unedited files (RAW: NEF / CR2 / CR3 / "
                            "ARW / DNG, or JPG / PNG). Paste the absolute path."
                        ),
                    )
                    edited_dir = gr.Textbox(
                        label="Edited folder",
                        placeholder=r"C:\path\to\edits",
                        info=(
                            "Folder containing the matching edited versions. Each file must "
                            "share its filename stem with one in the originals folder. "
                            "Extensions can differ (.NEF ↔ .jpg is fine)."
                        ),
                    )
                match_btn = gr.Button("Match Pairs")
                match_msg = gr.Textbox(label="Match result", interactive=False)
                unmatched_list = gr.JSON(label="Unmatched files")

                with gr.Row():
                    k_slider = gr.Slider(
                        3, 7, value=4, step=1,
                        label="Number of clusters",
                        info=(
                            "Groups your photos into N 'scene types' (sunny / golden hour / "
                            "indoor warm / blue hour / …); each cluster learns its own color "
                            "transform. Rule of thumb: aim for at least 20 pairs per cluster. "
                            "Use 3 if you shoot in one or two lighting conditions, 4 (default) "
                            "for a typical mixed library, 5-7 for very varied content."
                        ),
                    )
                    samples_slider = gr.Slider(
                        5000, 50000, value=10000, step=1000,
                        label="Pixel samples per pair",
                        info=(
                            "How many random pixels to sample from each before/after pair when "
                            "fitting the color transform. More samples = better generalization "
                            "but slower training. 10k (default) is the spec's recommendation; "
                            "raise to 30-50k if validation ΔE stays above 4."
                        ),
                    )
                with gr.Row():
                    val_slider = gr.Slider(
                        0.0, 0.4, value=0.2, step=0.05,
                        label="Validation fraction",
                        info=(
                            "Fraction of pairs held out of training and scored with CIEDE2000 "
                            "ΔE so you can see how close the learned style gets to your real "
                            "edits. ΔE<1 invisible, 1-2 barely perceptible, 3-5 acceptable, "
                            ">5 visibly off. Set to 0 if you have fewer than ~30 pairs."
                        ),
                    )
                    use_rf_check = gr.Checkbox(
                        value=True,
                        label="Train Random Forest (Method B)",
                        info=(
                            "Trains a per-pixel Random Forest on top of the Reinhard+curves "
                            "baseline. Catches local nonlinear edits (warm only highlights, "
                            "shift greens toward teal, dodge subject, etc.) that the baseline "
                            "can't model. Keep ON for libraries of 50+ pairs with selective "
                            "edits; turn OFF for tiny libraries (RF overfits) or pure global "
                            "looks (baseline already wins and profile stays smaller/faster)."
                        ),
                    )
                profile_name_box = gr.Textbox(
                    label="Profile name",
                    placeholder="my_style",
                    info=(
                        "Used as the filename for the saved profile (my_style.pkl + "
                        "my_style.json). No spaces recommended; existing profiles with the "
                        "same name will be overwritten."
                    ),
                )
                train_btn = gr.Button("Train Profile", variant="primary")
                train_status = gr.Textbox(label="Status", lines=4, interactive=False)
                cluster_plot = gr.Plot(label="Cluster assignments")

            # ---------- Apply ----------
            with gr.Tab("Apply"):
                with gr.Row():
                    apply_profile_dd = gr.Dropdown(
                        choices=list_profiles(profiles_dir), label="Profile"
                    )
                    apply_refresh = gr.Button("⟳ Refresh", scale=0)
                apply_input = gr.File(
                    label="Input image (JPG / PNG / RAW: NEF / CR2 / CR3 / ARW / DNG / ...)",
                    type="filepath",
                    file_count="single",
                    file_types=[
                        "image",
                        ".nef", ".cr2", ".cr3", ".arw", ".dng",
                        ".raf", ".rw2", ".orf", ".pef",
                    ],
                )
                apply_use_rf = gr.Checkbox(value=True, label="Use Random Forest (Method B)")
                apply_btn = gr.Button("Apply Profile", variant="primary")
                gr.Markdown(
                    "_Side-by-side before/after below. Click either image to see it full size._"
                )
                with gr.Row():
                    before_img = gr.Image(label="Before", type="numpy", interactive=False)
                    after_img = gr.Image(label="After", type="numpy", interactive=False)
                apply_status = gr.Textbox(label="Status", interactive=False)
                apply_download = gr.File(label="Download styled JPEG")

            # ---------- Quick Auto-Enhance ----------
            with gr.Tab("Quick Auto-Enhance"):
                gr.Markdown(
                    "**Independent from trained profiles.** Classical auto-correction "
                    "(white balance, exposure, shadows, highlights, local contrast, saturation). "
                    "Works on JPG, PNG, and RAW (NEF / CR2 / CR3 / ARW / DNG / …). "
                    "No training data required."
                )
                ae_input = gr.File(
                    label="Input image (JPG / PNG / RAW)",
                    type="filepath",
                    file_count="single",
                    file_types=[
                        "image",
                        ".nef", ".cr2", ".cr3", ".arw", ".dng",
                        ".raf", ".rw2", ".orf", ".pef",
                    ],
                )
                gr.Markdown(
                    "Click **Auto-Detect** to analyze the photo and set sensible defaults, "
                    "or move the sliders manually."
                )
                with gr.Row():
                    ae_wb = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.05,
                        label="White balance",
                        info="Gray-world correction. 0 = no change; 1 = full neutral correction.",
                    )
                    ae_exposure = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="Exposure",
                        info="Pulls average brightness toward neutral. Works in both directions (brightens dark shots, darkens overexposed ones).",
                    )
                with gr.Row():
                    ae_shadows = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="Shadows",
                        info="Lifts dark values without touching midtones or highlights.",
                    )
                    ae_highlights = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="Highlights",
                        info="Recovers bright values (compresses highlights). 1 = aggressive shoulder.",
                    )
                with gr.Row():
                    ae_local = gr.Slider(
                        0.0, 1.0, value=0.2, step=0.05,
                        label="Local contrast",
                        info="CLAHE on the L channel. Adds punch to flat photos. Too high = halos.",
                    )
                    ae_sat = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="Saturation",
                        info="Boosts HSV saturation. 1 ≈ +60%. Use sparingly on portraits.",
                    )
                with gr.Row():
                    ae_detect_btn = gr.Button("Auto-Detect", scale=1)
                    ae_apply_btn = gr.Button("Apply", variant="primary", scale=2)
                gr.Markdown("_Side-by-side before/after below. Click either image to zoom._")
                with gr.Row():
                    ae_before = gr.Image(label="Before", type="numpy", interactive=False)
                    ae_after = gr.Image(label="After", type="numpy", interactive=False)
                ae_status = gr.Textbox(label="Status", interactive=False)
                ae_download = gr.File(label="Download enhanced JPEG")

            # ---------- Export LUT ----------
            with gr.Tab("Export LUT"):
                with gr.Row():
                    lut_profile_dd = gr.Dropdown(
                        choices=list_profiles(profiles_dir), label="Profile"
                    )
                    lut_refresh = gr.Button("⟳ Refresh", scale=0)
                lut_cluster = gr.Dropdown(
                    choices=["average"], value="average", label="Cluster"
                )
                with gr.Row():
                    lut_size = gr.Slider(
                        17, 65, value=DEFAULT_LUT_SIZE, step=4, label="LUT size"
                    )
                    lut_use_rf = gr.Checkbox(value=True, label="Bake Method B (RF)")
                lut_btn = gr.Button("Generate .cube", variant="primary")
                lut_download = gr.File(label="Download .cube")
                lut_status = gr.Textbox(label="Status", interactive=False)
                gr.Markdown(
                    "### Use the LUT in Lightroom Classic\n"
                    "Lightroom can't read `.cube` directly. Two paths:\n\n"
                    "**Quick visual test (Photoshop):**\n"
                    "1. Open your photo in Photoshop.\n"
                    "2. `Image → Adjustments → Color Lookup`.\n"
                    "3. `Load 3D LUT…` → select the `.cube` you just downloaded.\n\n"
                    "**Permanent Lightroom profile:**\n"
                    "1. Wrap the `.cube` as a Camera Raw profile (`.xmp`). Free tool: "
                    "[lutbaker](https://github.com/cawtekfilms/lutbaker).\n"
                    "2. Drop the `.xmp` into:\n"
                    "    - Windows: `%APPDATA%\\Adobe\\CameraRaw\\Settings\\`\n"
                    "    - macOS: `~/Library/Application Support/Adobe/CameraRaw/Settings/`\n"
                    "3. Restart Lightroom → Develop → Profile → Browse → User Profiles.\n\n"
                    "**DaVinci Resolve** reads `.cube` natively: drop into the LUTs folder "
                    "and apply from the Color page."
                )

        # ---------- Event wiring ----------

        def _on_match(orig: str, ed: str) -> tuple[str, list[str]]:
            msg, unmatched = helper_match_pairs(orig, ed)
            return msg, unmatched

        def _on_train(
            orig: str,
            ed: str,
            k: float,
            samples: float,
            val: float,
            use_rf: bool,
            name: str,
            progress=gr.Progress(),
        ):
            def _cb(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            status, feats, labels = helper_train(
                orig,
                ed,
                int(k),
                int(samples),
                float(val),
                bool(use_rf),
                name,
                progress=_cb,
                profiles_dir=profiles_dir,
            )
            plot = _cluster_plot(feats, labels) if feats is not None else None
            new_choices = list_profiles(profiles_dir)
            return (
                status,
                plot,
                gr.update(choices=new_choices),
                gr.update(choices=new_choices),
            )

        def _on_apply(profile_name: str, image_path: str, use_rf: bool):
            before, after, dl, status = helper_apply(
                profile_name, image_path, use_rf, profiles_dir=profiles_dir
            )
            return before, after, dl, status

        def _on_export_lut(
            profile_name: str, cluster_choice: str, size: float, use_rf: bool
        ):
            path, status = helper_export_lut(
                profile_name, cluster_choice, int(size), bool(use_rf), profiles_dir=profiles_dir
            )
            return path, status

        def _on_lut_profile_change(profile_name: str):
            choices = _cluster_choices_for(profile_name, profiles_dir=profiles_dir)
            return gr.update(choices=choices, value=choices[0])

        def _refresh_all():
            choices = list_profiles(profiles_dir)
            return gr.update(choices=choices), gr.update(choices=choices)

        match_btn.click(
            _on_match, inputs=[originals_dir, edited_dir], outputs=[match_msg, unmatched_list]
        )
        train_btn.click(
            _on_train,
            inputs=[
                originals_dir,
                edited_dir,
                k_slider,
                samples_slider,
                val_slider,
                use_rf_check,
                profile_name_box,
            ],
            outputs=[train_status, cluster_plot, apply_profile_dd, lut_profile_dd],
        )
        apply_btn.click(
            _on_apply,
            inputs=[apply_profile_dd, apply_input, apply_use_rf],
            outputs=[before_img, after_img, apply_download, apply_status],
        )
        apply_refresh.click(_refresh_all, outputs=[apply_profile_dd, lut_profile_dd])
        lut_refresh.click(_refresh_all, outputs=[apply_profile_dd, lut_profile_dd])
        lut_profile_dd.change(
            _on_lut_profile_change, inputs=lut_profile_dd, outputs=lut_cluster
        )
        lut_btn.click(
            _on_export_lut,
            inputs=[lut_profile_dd, lut_cluster, lut_size, lut_use_rf],
            outputs=[lut_download, lut_status],
        )

        # Quick Auto-Enhance wiring
        def _on_ae_detect(image_path: str):
            return helper_auto_analyze(image_path)

        def _on_ae_apply(
            image_path: str,
            wb: float,
            exposure: float,
            shadows: float,
            highlights: float,
            local_c: float,
            saturation: float,
        ):
            return helper_auto_enhance(
                image_path, wb, exposure, shadows, highlights, local_c, saturation
            )

        ae_detect_btn.click(
            _on_ae_detect,
            inputs=ae_input,
            outputs=[ae_wb, ae_exposure, ae_shadows, ae_highlights, ae_local, ae_sat, ae_status],
        )
        ae_apply_btn.click(
            _on_ae_apply,
            inputs=[ae_input, ae_wb, ae_exposure, ae_shadows, ae_highlights, ae_local, ae_sat],
            outputs=[ae_before, ae_after, ae_download, ae_status],
        )

        return demo


def launch(
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
) -> int:
    """Build and launch the UI. Returns 0 on normal shutdown."""
    demo = build_ui(profiles_dir=profiles_dir)
    logger.info("Launching Gradio UI...")
    demo.launch(server_name=server_name, server_port=server_port, share=share)
    return 0
