# photo_style_ai

Learn your personal photo editing style from a folder of before/after pairs, then apply it to new photos. Export the learned style as a **`.cube`** 3D LUT for Lightroom, Photoshop, or DaVinci Resolve.

Runs entirely **locally on CPU**. No GPU, no cloud, no external APIs.

- **Input**: matched pairs of originals (RAW: `.NEF`, `.CR2`, `.CR3`, `.ARW`, `.DNG`, …, or `.JPG`/`.PNG`) and your edited versions
- **Output**: a `.pkl` style profile + `.json` manifest + downloadable `.cube` LUT
- **UI**: browser-based via Gradio, with a CLI for headless use

---

## How it works (in one paragraph)

Per-photo features (LAB brightness/contrast/color moments + histograms) are clustered with K-Means to group pairs by lighting/scene. For each cluster, two color transforms are learned:

- **Method A (baseline)**: Reinhard mean/std transfer in LAB + a per-channel polynomial tone curve fit on the residual.
- **Method B (refined)**: a multi-output Random Forest mapping `LAB pixel → LAB pixel`, trained on ~10k sampled pixels per pair.

A held-out validation set scores both methods with **CIEDE2000** so you can see how close the learned style gets to your real edits. The transform is also baked into a 33×33×33 `.cube` LUT for use in other tools.

---

## Requirements

- Python **3.10+**
- Windows, macOS, or Linux
- ~1 GB RAM for training on ~100 pairs
- A few hundred MB of disk for the feature cache + profiles

---

## Install

```bash
git clone <this repo>
cd photo-editing-style-replica
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -e .
```

On first run the UI will pull a couple of small Gradio assets. Everything else is local.

---

## Quick start

### 1. Prepare your pairs

Put originals and edits in **two folders** with **matching filename stems**. The extensions can differ — that's how a RAW pairs with a JPEG export.

```
NEFs/                       edits/
├── DSC_8701.NEF            ├── DSC_8701.jpg
├── DSC_8702.NEF            ├── DSC_8702.jpg
├── DSC_8703.NEF            ├── DSC_8703.jpg
└── ...                     └── ...
```

When exporting edited JPEGs from Lightroom, set the export filename to **Original filename** (not "Custom name with suffix") so stems match automatically.

How many pairs? Method A starts giving sensible results around **20+ pairs**. Method B (Random Forest) really shines from **~100 pairs** upward.

### 2. Launch the UI

```bash
python app.py
```

Open <http://127.0.0.1:7860>.

### 3. Train

1. **Train tab** → paste the two folder paths → click **Match Pairs** to verify the count.
2. Adjust **clusters** (3–7) and **samples per pair** (5k–50k — 10k is a good default).
3. Enter a **profile name** (no spaces preferred) → click **Train Profile**.
4. Watch the progress bar and final ΔE numbers in the status box:
   ```
   ΔE2000 over 20 val pairs: identity=8.30 methodA=3.10 methodB=2.40
   ```
   Lower ΔE = closer to your real edits. Anything below ~3 looks visually faithful; below ~2 is hard to distinguish from the original edit.

### 4. Apply to a new photo

**Apply tab** → pick the profile → upload a photo (JPG or RAW) → click **Apply Profile**. Side-by-side Before/After appears; **Download** writes a JPEG at quality 95.

### 5. Export a LUT

**Export LUT tab** → pick the profile → choose **Cluster** (`average` or a specific cluster index) → **Generate .cube** → download.

Filename convention: `<profile>_<cluster>.cube` (e.g. `my_style_average.cube`).

---

## Using the LUT in Lightroom Classic

Lightroom Classic **cannot read `.cube` directly** — it needs an `.xmp` Camera Raw profile. Two reliable paths:

### A. Quick visual test (Photoshop, recommended first)

1. Open the photo in Photoshop.
2. `Image → Adjustments → Color Lookup`.
3. `Load 3D LUT…` → select the `.cube`.

If the look matches your style here, the LUT is correct. Move on to wrapping it for Lightroom.

### B. Install permanently in Lightroom Classic

1. Wrap the `.cube` as a Camera Raw profile (`.xmp`). Free tool: [lutbaker](https://github.com/cawtekfilms/lutbaker).
2. Drop the resulting `.xmp` into the Camera Raw settings folder:
   - **Windows**: `%APPDATA%\Adobe\CameraRaw\Settings\`
   - **macOS**: `~/Library/Application Support/Adobe/CameraRaw/Settings/`
3. Restart Lightroom → **Develop → Profile → Browse → User Profiles**. Your profile appears alongside Adobe Standard, Camera Neutral, etc.

### Other tools

- **DaVinci Resolve**: drop the `.cube` into your LUT folder, then Color page → right-click clip → Apply LUT.
- **Adobe Camera Raw Filter** (in Photoshop): same `.xmp` wrapping path as Lightroom.

---

## CLI reference

The UI is built on the same CLI. Use it for headless training, scripting, or running on a remote box.

```bash
# Train (same defaults as the UI)
python app.py --no-ui train \
    --originals path/to/NEFs \
    --edited path/to/edits \
    --name my_style \
    --k 4 \
    --samples-per-pair 10000 \
    --val-fraction 0.2

# Apply
python app.py --no-ui apply \
    --profile my_style \
    --input shot.NEF \
    --output styled.jpg

# Export a 33^3 LUT
python app.py --no-ui export-lut \
    --profile my_style \
    --output my_style_avg.cube \
    --cluster average

# List saved profiles
python app.py --no-ui list
```

Skip Method B entirely (Method A only) with `--no-rf` on `train` / `apply` / `export-lut`. Useful for small profile size or when validation shows Method A already wins.

The `photo-style-ai` console script (set up by `pyproject.toml`) is equivalent to `python app.py --no-ui`.

---

## Profile artifacts

```
profiles/
├── my_style.pkl     # joblib-compressed: cluster model + per-cluster transforms
└── my_style.json    # human-readable manifest
```

Manifest example:
```json
{
  "name": "my_style",
  "created_at": "2026-05-21T19:15:16+00:00",
  "photo_count": 120,
  "cluster_count": 4,
  "samples_per_pair": 10000,
  "feature_max_dim": 512,
  "schema_version": 1,
  "validation_delta_e": 2.40,
  "extra": {
    "validation_metrics": {
      "n_pairs": 24, "identity": 8.30, "method_a": 3.10, "method_b": 2.40
    }
  }
}
```

Profiles are small — typically 1–25 MB with default settings, well under the 50 MB cap.

The feature cache lives at `.cache/features.pkl` and is keyed by `(absolute path, mtime, size)`. Touching a file invalidates its entry, otherwise retraining is near-instant on the feature-extraction step.

---

## Performance expectations

| Workload | Time on a recent CPU |
|---|---|
| Feature extraction (cached after first run) | ~50 ms / photo |
| Clustering | <1 s |
| Per-cluster Method A fit | ~50 ms |
| Per-cluster Method B (RF) fit, defaults | ~5 s |
| Validation (Method A + B) on 20 pairs | ~10 s |
| **Train, 100 pairs, k=4, RF on** | **~2–4 minutes** |
| Apply Method A to a 24 MP RAW | ~3 s |
| Apply Method B (RF) to a 24 MP RAW | ~30–60 s (it's per-pixel forest predict) |
| Apply via exported .cube in Lightroom/DaVinci | sub-second (LUT interpolation) |

For full-res inference on RAW, the `.cube` LUT path is dramatically faster than running the RF directly. Train once, export, then use the LUT.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'loguru'` (or `gradio`, `rawpy`, …)
You haven't installed deps. From the project root:
```bash
python -m pip install -e .
```

### "Found 0 matched pair(s)"
Filename stems aren't matching. The CLI prints which files are unmatched on each side:
```
WARNING Unmatched original: IMG_001_edited.jpg
WARNING Unmatched edit: IMG_001.jpg
```
Fix by renaming so stems match. Matching is case-insensitive; extensions can differ (`.NEF` ↔ `.jpg` is fine).

### NEF (or other RAW) loads but colors look wrong / image is dark
Check `rawpy` is installed (`pip show rawpy`). The demosaic uses camera white balance + auto-brightness + standard sRGB gamma. If results are wildly off:
- Make sure your **edited** file is actually the edit (not the in-camera JPEG that ships alongside the NEF). The in-camera JPEG bakes in Nikon's default rendering, not your style.
- Pair at least 30+ shots before judging. Single-pair training is intentionally crude — the spec stacks multiple methods to fix that with more data.

### "Subject is too dark / too flat" after applying the profile
Two common causes:
1. **Too few pairs** — global Reinhard can't tell subject from background. Method B (RF) starts learning local behavior around 100 pairs.
2. **Cluster mismatch** — your input photo got assigned to the wrong cluster. In the **Export LUT** tab, generate per-cluster `.cube` files (Cluster 0, 1, 2, …) and visually pick the one that matches. Use that cluster index with `apply --cluster N` for similar shots.

### Method A beats Method B in validation
That's fine — and informative. It means your editing style is mostly a smooth global transform (warmth + contrast curve), which polynomial curves model perfectly. Random Forest needs enough data to learn local nonlinear behavior. Either:
- Train on more pairs, or
- Stick with `--no-rf` for a smaller, faster profile

### Profile is over 50 MB
Random Forest is the only thing that grows large. Lower one or more of:
```
--rf-trees 20         # default 40
--rf-max-depth 8      # default 10
--rf-max-samples 40000 # default 80000
```
Each ~halves the forest's contribution to file size.

### Method B inference is slow on full-res RAW
Expected — the RF predicts pixel by pixel. Workarounds:
- Apply Method A (`--no-rf`) for fast previews
- **Export a `.cube` LUT** and apply via Lightroom/DaVinci/Photoshop — sub-second on any size

### Lightroom doesn't see the `.cube`
Lightroom never reads `.cube` directly. You must wrap as `.xmp` first — see the [Lightroom section](#using-the-lut-in-lightroom-classic) above.

### Gradio UI doesn't open / port conflict
Gradio defaults to port 7860. If something else is using it:
```bash
python -c "from photo_style.ui import launch; launch(server_port=7870)"
```

### Tests
Run the suite to confirm your install is working:
```bash
python -m pytest tests/ -v
```
Expect 61 tests passing.

---

## Design constraints (per the original spec)

- All math runs in **LAB** color space (perceptually uniform, not RGB)
- RAW files are demosaiced to **16-bit linear** then converted to sRGB
- Profiles are self-contained `.pkl` + `.json` (no full image data stored)
- **EXIF is intentionally preserved by Pillow on JPEG re-save**; auto-rotate / auto-crop are not applied
- No PyTorch / TensorFlow / external APIs — only NumPy, SciPy, scikit-learn, scikit-image, colour-science, OpenCV, rawpy, Gradio

---

## Project layout

```
photo-editing-style-replica/
├── app.py                          # Gradio + --no-ui entry
├── pyproject.toml
├── README.md
├── photo_style/
│   ├── io_utils.py                 # load/save, RAW, pair discovery
│   ├── features.py                 # per-photo LAB features
│   ├── feature_cache.py            # pickle cache keyed on path+mtime
│   ├── clustering.py               # StandardScaler + K-Means
│   ├── style_model.py              # Reinhard + curves + RF
│   ├── inference.py                # apply_profile()
│   ├── evaluation.py               # CIEDE2000 + validation pipeline
│   ├── profile_store.py            # save/load profile + manifest
│   ├── lut_export.py               # 33^3 .cube generation
│   ├── cli.py                      # train / apply / export-lut / list
│   └── ui.py                       # Gradio Blocks + helpers
├── tests/                          # 61 tests
└── profiles/                       # saved profiles (gitignored)
```

---

## License

MIT.
