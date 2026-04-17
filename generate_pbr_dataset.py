"""
generate_pbr_dataset.py
───────────────────────────────────────────────────────────────────────────────
Renders a dataset of randomised PBR materials from a Substance Designer .sbsar
file using the sbsrender CLI.

Output structure
─────────────────
PBR_Dataset/
├── dataset.csv                   ← master index of every rendered set
└── renders/
    ├── 042817/                   ← folder name = 6-digit random seed
    │   ├── 042817_basecolor.png
    │   ├── 042817_normal.png
    │   ├── 042817_roughness.png
    │   └── ...
    └── 391054/
        └── ...

Usage
──────
Just run:  python generate_pbr_dataset.py
Stop at any time with Ctrl-C — already-completed seeds are skipped on re-run.
"""

import math
import os
import csv
import glob
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Add the PBR renderer to the import path ──────────────────────────────────
PBR_RENDERER_PATH = r"F:\!_GitHub_Rep\PBR-Python-Shaderball-Renderer"
if PBR_RENDERER_PATH not in sys.path:
    sys.path.insert(0, PBR_RENDERER_PATH)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       USER-CONFIGURABLE SETTINGS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── How many unique material sets to generate ─────────────────────────────────
NUM_MATERIALS = 1000

# ── Path to your .sbsar file ──────────────────────────────────────────────────
SBSAR_PATH = r"F:\!_GitHub_Rep\PBR_Training\random-material.sbsar"

# ── Root output folder (CSV lives here; renders/ subfolder is created inside) ─
OUTPUT_ROOT = r"F:\!_GitHub_Rep\PBR_Training\PBR_Dataset"

# ── sbsrender executable ──────────────────────────────────────────────────────
#    Common install locations — the script will auto-detect the first one found.
#    Add or reorder entries if yours is somewhere else.
SBSRENDER_CANDIDATES = [
    r"C:\Program Files\Adobe\Adobe Substance 3D Designer\sbsrender.exe",
    r"C:\Program Files\Allegorithmic\Substance Designer\sbsrender.exe",
    r"C:\Program Files\Adobe\Adobe Substance Designer\sbsrender.exe",
    # macOS / Linux fallbacks
    "/Applications/Adobe Substance 3D Designer.app/Contents/MacOS/sbsrender",
    "/usr/local/bin/sbsrender",
]

# ── Output image format: "png" | "tga" | "exr" | "tif" ───────────────────────
OUTPUT_FORMAT = "png"

# ── Output texture resolution (must be a power of 2) ─────────────────────────
OUTPUT_RESOLUTION = 512   # e.g. 512, 1024, 2048, 4096

# ── Sphere renderball preview ─────────────────────────────────────────────────
SPHERE_RENDER_SIZE  = 512              # pixel resolution of the renderball image
DISPLACEMENT_SCALE  = 10.0             # height displacement intensity
HDR_ENVIRONMENT_PATH = ""              # optional default HDR/EXR environment map;
                                       # leave blank to auto-use the "environment"
                                       # channel from sbsrender or a procedural sky
GPU_RENDER_SAMPLES  = 256              # path-tracing samples per pixel (more = cleaner)
GPU_MAX_BOUNCES     = 6                # max light bounces for path tracer
CAMERA_DISTANCE     = 23.4             # camera distance from shaderball (cm)
CAMERA_FOV          = 23.67            # vertical field of view (degrees)

# ── Auto-exposure settings ────────────────────────────────────────────────────
AUTO_EXPOSURE_MIN   = 0.005            # minimum exposure clamp
AUTO_EXPOSURE_MAX   = 50.0             # maximum exposure clamp
AUTO_EXPOSURE_BIAS  = 0.0              # bias in log2 stops (positive = brighter)

# ── Texture tiling ────────────────────────────────────────────────────────────
TEXTURE_TILING      = 5.0              # UV tile repeat for surface PBR textures

# ── Renderball viewing angles ─────────────────────────────────────────────────
#    Each entry defines a camera orbit angle around the vertical axis.
#    azimuth_deg: 0 = front-right (matching StandardShaderBall default ~20°).
#    Three views at 30° horizontal offsets.
RENDERBALL_ANGLES = [
    {"label": "front",  "azimuth_deg": 25, "elevation_deg": 38},
    {"label": "left",   "azimuth_deg": -5, "elevation_deg": 38},
    {"label": "right",  "azimuth_deg": 55, "elevation_deg": 38},
]

# ── Tessellation / Displacement note ──────────────────────────────────────────
#    sbsrender is a 2D texture baker — it does NOT support 3D scene rendering,
#    tessellation, or geometric displacement.  Those features belong to the 3D
#    viewport inside Substance Designer (Iray) or a separate 3D app.
#
#    The desired 3D viewport settings (for reference only):
#       Height Scale        = 10
#       Tessellation Factor = 16
#       Phong Tessellation  = True
#
#    Displacement is instead applied in the Python sphere renderer below,
#    which modifies the sphere geometry with the baked height map.

# ── PBR channel names that sbsrender is expected to produce ──────────────────
#    These are matched against the filenames sbsrender writes out.
#    The script does a case-insensitive search so slight naming differences are
#    handled automatically (e.g. "BaseColor" vs "basecolor").
CHANNEL_NAMES = [
    "basecolor",
    "normal",
    "roughness",
    "metallic",
    "height",
    "ambientocclusion",
    "specular",
    "glossiness",
    "background",
    "environment"
]

# ── CSV column order ──────────────────────────────────────────────────────────
CSV_COLUMNS = ["seed", "folder_path"] + CHANNEL_NAMES + \
              [f"renderball_{a['label']}" for a in RENDERBALL_ANGLES]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          INTERNAL HELPERS                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_sbsrender() -> str:
    """Return the path to sbsrender, checking candidates then PATH."""
    for candidate in SBSRENDER_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    # Fall back to PATH
    found = shutil.which("sbsrender")
    if found:
        return found
    return ""


def validate_paths(sbsrender_exe: str) -> None:
    """Abort early with a clear message if key paths are missing."""
    errors = []
    if not sbsrender_exe:
        errors.append(
            "sbsrender.exe not found.\n"
            "  → Add its full path to SBSRENDER_CANDIDATES at the top of this script."
        )
    if not os.path.isfile(SBSAR_PATH):
        errors.append(
            f"SBSAR file not found: {SBSAR_PATH}\n"
            "  → Update SBSAR_PATH at the top of this script."
        )
    if errors:
        print("\n── Configuration errors ─────────────────────────────────")
        for e in errors:
            print(f"  ✗ {e}")
        print()
        sys.exit(1)


def generate_seed() -> str:
    """Return a 6-digit zero-padded random seed string, e.g. '042817'."""
    return f"{random.randint(0, 999999):06d}"


def seed_folder(seed: str) -> Path:
    return Path(OUTPUT_ROOT) / "renders" / seed


def render_material(sbsrender_exe: str, seed: str) -> bool:
    """
    Call sbsrender for one seed.
    Returns True on success, False on failure.
    """
    out_dir = seed_folder(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sbsrender_exe,
        "render",
        "--inputs",        SBSAR_PATH,
        "--output-path",   str(out_dir),
        "--output-name",   f"{seed}_{{outputNodeName}}",
        "--output-format", OUTPUT_FORMAT,
        "--set-value",     f"$outputsize@{_res_to_sbsrender(OUTPUT_RESOLUTION)}",
        "--set-value",     f"$randomseed@{int(seed)}",
    ]

    print(f"  ▶ sbsrender  seed={seed}  res={OUTPUT_RESOLUTION}  fmt={OUTPUT_FORMAT}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,          # 5-minute timeout per render
        )
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out (seed {seed})")
        return False
    except FileNotFoundError:
        print(f"  ✗ sbsrender not found at: {sbsrender_exe}")
        return False

    if result.returncode != 0:
        print(f"  ✗ sbsrender exited with code {result.returncode}")
        if result.stderr:
            for line in result.stderr.splitlines()[:10]:
                print(f"     {line}")
        return False

    # ── Re-render environment channel as HDR for proper HDRI lighting ─────
    hdr_cmd = [
        sbsrender_exe,
        "render",
        "--inputs",        SBSAR_PATH,
        "--output-path",   str(out_dir),
        "--output-name",   f"{seed}_{{outputNodeName}}",
        "--output-format", "hdr",
        "--set-value",     f"$outputsize@{_res_to_sbsrender(OUTPUT_RESOLUTION)}",
        "--set-value",     f"$randomseed@{int(seed)}",
    ]
    try:
        hdr_result = subprocess.run(
            hdr_cmd, capture_output=True, text=True, timeout=300,
        )
        if hdr_result.returncode == 0:
            # Keep only the environment .hdr (HDRI lighting), remove others
            for hdr_f in out_dir.glob("*.hdr"):
                stem_lower = hdr_f.stem.lower()
                if "environment" not in stem_lower:
                    hdr_f.unlink()
    except Exception:
        pass  # HDR re-render is best-effort; PNG environment still works

    return True


def _res_to_sbsrender(res: int) -> str:
    """
    Convert a pixel resolution to the Substance $outputsize format.
    sbsrender uses log2 integers: 512→9, 1024→10, 2048→11, 4096→12
    The value is passed as "W,H" where both W and H use the log2 form.
    """
    exp = int(math.log2(res))
    return f"{exp},{exp}"


def discover_channels(seed: str) -> dict[str, str]:
    """
    Scan the seed folder and map each recognised channel name to its file path.
    Returns a dict: { "basecolor": "...\\042817_basecolor.png", ... }
    Missing channels map to an empty string.
    Prefers .hdr over .png for the environment channel (better HDRI lighting).
    """
    folder = seed_folder(seed)
    # Collect all output files (png + hdr)
    files = list(folder.glob(f"*.{OUTPUT_FORMAT}")) + list(folder.glob("*.hdr"))

    channel_map = {ch: "" for ch in CHANNEL_NAMES}

    for f in files:
        stem_norm = f.stem.lower().replace("_", "")
        for channel in CHANNEL_NAMES:
            if channel.replace("_", "") in stem_norm:
                # Prefer HDR for environment channel (used as HDRI lighting)
                if channel == "environment" and f.suffix.lower() == ".hdr":
                    channel_map[channel] = str(f)
                elif not channel_map[channel] or \
                     (channel != "environment"):
                    channel_map[channel] = str(f)
                break

    return channel_map


def load_existing_csv(csv_path: Path) -> tuple[list[dict], set[str]]:
    """
    Load previously written rows from the CSV.
    Returns (rows_list, set_of_existing_seeds).
    """
    rows, seeds = [], set()
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
                seeds.add(row.get("seed", ""))
    return rows, seeds


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write the full dataset CSV, creating/overwriting the file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(total: int, succeeded: int, skipped: int, failed: int,
                  elapsed: float) -> None:
    mins, secs = divmod(int(elapsed), 60)
    print()
    print("── Summary ──────────────────────────────────────────────────")
    print(f"  Requested  : {total}")
    print(f"  Rendered   : {succeeded}  ✓")
    print(f"  Skipped    : {skipped}  (already existed)")
    print(f"  Failed     : {failed}  ✗")
    print(f"  Time       : {mins}m {secs}s")
    print("─────────────────────────────────────────────────────────────")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║               GPU PBR SHADERBALL RENDERER (path-traced)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_gpu_pt = None  # singleton GPUPathtracer, lazily initialised


def _init_gpu_renderer():
    """Lazy-init the GPU path tracer (called once on first use)."""
    global _gpu_pt
    if _gpu_pt is not None:
        return _gpu_pt

    from python_renderer.core.material import OpenPBRMaterial, RenderConfig
    from python_renderer.gpu.pathtracer import GPUPathtracer

    # StandardShaderBall: units are cm, sphere Ø7.53cm
    # Camera matches OpenPBR-viewer: FOV≈23.67°, distance≈23.4cm, elevation≈40°
    config = RenderConfig(
        width=SPHERE_RENDER_SIZE,
        height=SPHERE_RENDER_SIZE,
        samples=GPU_RENDER_SAMPLES,
        max_bounces=GPU_MAX_BOUNCES,
        use_gpu=True,
        camera_distance=CAMERA_DISTANCE,
        camera_fov=CAMERA_FOV,
    )

    # Default white material — textures will override per-render
    material = OpenPBRMaterial(
        base_color=(0.5, 0.5, 0.5),
        base_weight=1.0,
        base_metalness=0.0,
        specular_weight=1.0,
        specular_roughness=0.5,
        specular_ior=1.5,
    )

    # Paths to StandardShaderBall assets in PBR-Python-Shaderball-Renderer
    assets_dir = os.path.join(PBR_RENDERER_PATH, "assets")
    sb_dir = os.path.join(assets_dir, "standard-shader-ball")
    # Prefer .obj (user-unwrapped UVs) over .glb (partial UVs from OpenPBR)
    surface_obj = os.path.join(sb_dir, "openpbr_surface_full.obj")
    surface_glb = os.path.join(sb_dir, "openpbr_objects.glb")
    surface_mesh = surface_obj if os.path.isfile(surface_obj) else surface_glb
    neutral_glb = os.path.join(sb_dir, "neutral_objects.glb")
    ground_tex  = os.path.join(assets_dir, "textures", "ground.png")

    pt = GPUPathtracer(config)
    pt.setup_gl()
    pt._surface_tex_tile = TEXTURE_TILING
    pt.load_shaderball_scene(
        surface_glb=surface_mesh,
        neutral_glb=neutral_glb,
        material=material,
        ground_tex_path=ground_tex,
        ground_y=0.0,
    )

    _gpu_pt = pt
    return pt


def _load_tex_for_gpu(path: str) -> np.ndarray | None:
    """Load a texture image as float32 RGB(A) array, or None if missing."""
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def render_sphere_preview(channel_map: dict, seed: str,
                          azimuth_deg: float = 0.0,
                          elevation_deg: float = 38.0,
                          suffix: str = "",
                          env_override: str = "") -> str:
    """
    GPU path-traced PBR shaderball render with camera orbit and HDR environment.
    Returns the saved renderball file path.
    env_override: if set, use this HDR path instead of the channel_map environment.
    """
    pt = _init_gpu_renderer()

    # ── Upload PBR textures ───────────────────────────────────────────────
    _TEX_CHANNEL_MAP = {
        "basecolor":        "basecolor",
        "normal":           "normal",
        "roughness":        "roughness",
        "metallic":         "metallic",
        "ambientocclusion": "ao",
        "specular":         "specular",
        "height":           "height",
    }
    # Clear previous textures
    pt._pbr_textures.clear()

    for chan_name, gpu_name in _TEX_CHANNEL_MAP.items():
        tex = _load_tex_for_gpu(channel_map.get(chan_name, ""))
        if tex is not None:
            pt.load_pbr_texture(gpu_name, tex)

    # Glossiness → roughness inversion
    gloss_path = channel_map.get("glossiness", "")
    if gloss_path and os.path.isfile(gloss_path) and "roughness" not in pt._pbr_textures:
        gloss = _load_tex_for_gpu(gloss_path)
        if gloss is not None:
            pt.load_pbr_texture("roughness", 1.0 - gloss)

    # ── Load environment map (from environment channel or fallback HDR) ────
    if env_override and os.path.isfile(env_override):
        env_path = env_override
    else:
        env_path = channel_map.get("environment", "")
        if not env_path or not os.path.isfile(env_path):
            env_path = HDR_ENVIRONMENT_PATH
    if env_path and os.path.isfile(env_path):
        pt.load_environment(env_path)

        # ── Auto-exposure from HDRI luminance ─────────────────────────────
        try:
            import cv2
            hdr_img = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
            if hdr_img is None:
                raise RuntimeError(f"cv2 could not read {env_path}")
            hdr_img = hdr_img.astype(np.float32)
            # cv2 loads BGR; use BGR luminance weights (B=0.0722, G=0.7152, R=0.2126)
            lum = 0.0722 * hdr_img[..., 0] + 0.7152 * hdr_img[..., 1] + 0.2126 * hdr_img[..., 2]
            # Filter out near-black pixels that poison the log-average
            valid = lum > 0.01
            valid_frac = valid.sum() / lum.size
            if valid_frac < 0.05:
                # Fewer than 5% valid pixels — fall back to median
                log_avg = float(np.median(lum[lum > 0]) if (lum > 0).any() else 0.18)
            else:
                log_avg = np.exp(np.mean(np.log(lum[valid])))
            # Map log-average to calibrated key (0.6 accounts for indirect lighting)
            raw_exposure = 0.6 / max(log_avg, 1e-6)
            # Coverage correction: when only a fraction of the environment
            # provides significant light, total illumination is lower than
            # the log-average of bright pixels suggests.  Boost exposure to
            # compensate (capped to avoid extreme values).
            if valid_frac < 1.0:
                coverage_boost = 1.0 / max(valid_frac, 0.1)
                raw_exposure *= min(coverage_boost, 5.0)
            biased_exposure = raw_exposure * (2.0 ** AUTO_EXPOSURE_BIAS)
            pt._exposure = max(AUTO_EXPOSURE_MIN, min(AUTO_EXPOSURE_MAX, biased_exposure))
            print(f"  ▶ Auto-exposure: log_avg_lum={log_avg:.4f}  raw={raw_exposure:.4f}  coverage={valid_frac*100:.1f}%  final={pt._exposure:.4f}")
        except Exception as e:
            print(f"  ⚠ Auto-exposure failed ({e}), using default 1.0")
            pt._exposure = 1.0
    else:
        # No environment map — reset exposure to neutral default so stale
        # values from a previous seed's HDRI don't bleed into this render.
        pt._exposure = 1.0
        print(f"  ⚠ No environment map found — using default exposure 1.0")

    # ── Set camera orbit ──────────────────────────────────────────────────
    pt.set_camera_orbit(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)

    # ── Render ────────────────────────────────────────────────────────────
    image = pt.render()

    # ── Save ──────────────────────────────────────────────────────────────
    tag = f"_{suffix}" if suffix else ""
    out_path = seed_folder(seed) / f"{seed}_renderball{tag}.{OUTPUT_FORMAT}"
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(str(out_path))
    return str(out_path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                               MAIN                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    start_time = time.time()

    # ── Locate sbsrender ──────────────────────────────────────────────────────
    sbsrender_exe = find_sbsrender()
    validate_paths(sbsrender_exe)

    print(f"\n── PBR Dataset Generator ──────────────────────────────────────")
    print(f"  sbsar        : {SBSAR_PATH}")
    print(f"  sbsrender    : {sbsrender_exe}")
    print(f"  output root  : {OUTPUT_ROOT}")
    print(f"  materials    : {NUM_MATERIALS}")
    print(f"  resolution   : {OUTPUT_RESOLUTION}×{OUTPUT_RESOLUTION}")
    print(f"  format       : {OUTPUT_FORMAT}")
    print(f"───────────────────────────────────────────────────────────────\n")

    # ── Setup directories ─────────────────────────────────────────────────────
    renders_dir = Path(OUTPUT_ROOT) / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(OUTPUT_ROOT) / "dataset.csv"

    # ── Load any existing progress ────────────────────────────────────────────
    existing_rows, existing_seeds = load_existing_csv(csv_path)

    already_done = 0
    # Re-check existing seeds — verify the folder actually exists on disk
    for row in existing_rows:
        if not seed_folder(row["seed"]).is_dir():
            existing_seeds.discard(row["seed"])

    succeeded = 0
    failed    = 0
    skipped   = len(existing_seeds)
    all_rows  = list(existing_rows)  # start with whatever was already written

    remaining = NUM_MATERIALS - (len(existing_seeds))
    if remaining <= 0:
        print(f"✔ Dataset already has {len(existing_seeds)} renders. Nothing to do.")
        print(f"  (Increase NUM_MATERIALS if you want more.)\n")
        write_csv(csv_path, all_rows)
        return

    print(f"  {skipped} existing seeds found — skipping those.\n")

    generated_this_run = 0

    while generated_this_run < remaining:
        # Generate a unique seed
        attempts = 0
        while True:
            seed = generate_seed()
            if seed not in existing_seeds:
                break
            attempts += 1
            if attempts > 100_000:
                print("  ✗ Could not find a unique unused seed — dataset may be full.")
                sys.exit(1)

        folder = seed_folder(seed)
        progress = generated_this_run + 1
        total_so_far = len(existing_seeds) + progress
        print(f"[{total_so_far}/{NUM_MATERIALS}] Seed {seed}", end="")

        # Skip if the folder was somehow pre-created outside this script
        if folder.is_dir() and any(folder.iterdir()):
            print(f"  → folder already exists, skipping")
            existing_seeds.add(seed)
            skipped += 1
            generated_this_run += 1
            continue

        print()

        # ── Render ────────────────────────────────────────────────────────────
        ok = render_material(sbsrender_exe, seed)

        if not ok:
            print(f"  ✗ Render failed for seed {seed} — removing partial folder")
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
            failed += 1
            generated_this_run += 1
            continue

        # ── Discover output files ─────────────────────────────────────────────
        channels = discover_channels(seed)
        found_channels = [k for k, v in channels.items() if v]

        if not found_channels:
            print(f"  ✗ No output files found in {folder} — check sbsrender output names")
            failed += 1
            generated_this_run += 1
            continue

        print(f"  ✓ Channels: {', '.join(found_channels)}")

        # ── Render sphere previews (multiple angles) ─────────────────────
        rb_paths = {}
        for angle_cfg in RENDERBALL_ANGLES:
            label = angle_cfg["label"]
            try:
                p = render_sphere_preview(
                    channels, seed,
                    azimuth_deg=angle_cfg["azimuth_deg"],
                    elevation_deg=angle_cfg.get("elevation_deg", 38.0),
                    suffix=label,
                )
                rb_paths[label] = p
                print(f"  ✓ Renderball ({label}): {Path(p).name}")
            except Exception as exc:
                rb_paths[label] = ""
                print(f"  ⚠ Renderball ({label}) failed: {exc}")

        # ── Record row (paths relative to the CSV file's directory) ─────────
        csv_dir = csv_path.parent
        row = {
            "seed":        seed,
            "folder_path": str(folder.relative_to(csv_dir)),
            **{k: (str(Path(v).relative_to(csv_dir)) if v else "") for k, v in channels.items()},
            **{f"renderball_{lbl}": (str(Path(p).relative_to(csv_dir)) if p else "")
               for lbl, p in rb_paths.items()},
        }
        all_rows.append(row)
        existing_seeds.add(seed)

        # Write CSV after every successful render so progress is never lost
        write_csv(csv_path, all_rows)

        succeeded        += 1
        generated_this_run += 1

    # ── Final CSV write ───────────────────────────────────────────────────────
    write_csv(csv_path, all_rows)

    print(f"\n  CSV saved → {csv_path}")
    print_summary(NUM_MATERIALS, succeeded, skipped, failed, time.time() - start_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user — partial progress saved to CSV.")
        sys.exit(0)
