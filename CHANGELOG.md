# Changelog

All notable changes to the PBR Training Dataset Generator are documented here.
Entries are in reverse chronological order.

---

## [0.9.0] — 2026-04-17

### Added
- **Per-angle camera elevation** — Each renderball angle entry now has its own `elevation_deg` field (azimuth + elevation per view), replacing the former global `CAMERA_ELEVATION` setting.
- Camera Elevation° entry added to each angle row in the GUI.

### Removed
- Global `CAMERA_ELEVATION` setting from render settings section (moved to per-angle).

---

## [0.8.0] — 2026-04-17

### Added
- **Camera elevation GUI control** — Added `CAMERA_ELEVATION` global (default 38.0°) and corresponding GUI entry in the Render Settings section.
- Elevation fully wired into config persistence, presets, and auto-save.

---

## [0.7.0] — 2026-04-17

### Fixed
- **Auto-exposure coverage correction** — When only a fraction of the HDR environment provides significant illumination (e.g., Dark.hdr at 22.5% coverage), the auto-exposure now boosts inversely with coverage fraction (capped at 5×). This prevents dark environments from producing underexposed renders.
  - Before: Dark.hdr → exposure=12.35, mean brightness=0.12 (too dark)
  - After: Dark.hdr → exposure=50.0, mean brightness=0.32 (balanced)

---

## [0.6.0] — 2026-04-17

### Fixed
- **Critical HDR reading bug** — Discovered that `imageio` (v2 and v3) reads `.hdr` (RGBE) files as `uint8` (0–255) instead of `float32` radiance values, making all environment maps ~255× too bright.
  - Replaced `imageio.imread()` with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` in both `texture_loader.py` (renderer) and `generate_pbr_dataset.py` (auto-exposure).
  - Added BGR→RGB channel conversion for OpenCV's channel order.
  - Correct values verified: sbsrender env 0.007–1.0, Bright.hdr 0.001–6.5, Dark.hdr 0.000–3.8 (vs. imageio's broken 0–255).

### Added
- **Stale exposure reset** — When no environment map is found for a seed, `pt._exposure` is now reset to 1.0 with a warning, preventing exposure bleeding from a previous seed's HDRI.

---

## [0.5.0] — 2026-04-16

### Fixed
- **Buffer cleanup crash** — Wrapped `__del__` in `try/except` for both `GPUBuffer` and `TextureBuffer` in `buffer.py` to suppress `OSError: access violation reading` during Python shutdown (harmless GC-after-OpenGL-context-destroy).

### Reviewed
- **Exposure settings audit** — Comprehensive review of exposure parameters (MIN=0.005, MAX=50.0, BIAS=0.0, key=0.6). Confirmed correctness of auto-exposure pipeline.
- **Environment map pipeline audit** — Verified sbsrender runs twice (PNG for channels, HDR for environment), `discover_channels()` correctly prefers `.hdr` over `.png` for environment channel.

---

## [0.4.0] — 2026-04-15

### Fixed
- **VNDF sampling** — Implemented Dupuy & Benyoub 2023 bounded VNDF sampling in `common.glsl` with safe normalize fallback to geometric normal.
- **Specular BRDF denominator** — Corrected from `4*NdotV` to `4*NdotL*NdotV` in `pathtracer.glsl`.
- **VNDF PDF** — Fixed to `G1V * D / (4 * NdotV)` for correct importance sampling weight.
- **NaN/Inf defenses** — Added guards at accumulation and throughput levels in `pathtracer.glsl`, with early-exit on degenerate throughput.
- Mirror reflections now render correctly.

---

## [0.3.0] — 2026-04-14

### Added
- **GPU path tracer integration** — OpenGL 4.3 compute-shader path tracer with BVH acceleration, replacing the original software renderer.
- **StandardShaderBall model** — User-unwrapped `.obj` mesh (24,039 vertices, 46,408 faces) for realistic material previews.
- **ACES tonemapping** — Filmic tonemapping with configurable exposure applied in Python after raw HDR accumulation buffer readback.
- **Multi-angle rendering** — `RENDERBALL_ANGLES` config with front (25°), left (-5°), and right (55°) camera orbits.
- **Texture tiling** — Configurable UV tile repeat (default 5.0×) for PBR surface textures.

### Changed
- Camera parameters tuned to match OpenPBR viewer: distance 23.4cm, FOV 23.67°, elevation 38°.

---

## [0.2.0] — 2026-04-13

### Added
- **Tkinter GUI** (`pbr_dataset_gui.py`) — Full desktop interface for the dataset generator.
  - Render settings: resolution, GPU samples, max bounces, texture tiling, camera distance/FOV.
  - Auto-exposure controls: min/max exposure clamps, log2 bias.
  - HDR environment override path with file browser.
  - Output channel checkboxes (basecolor, normal, roughness, metallic, height, AO, specular, glossiness, background, environment).
  - Renderball angle editor: add/remove rows with label + azimuth.
  - **Preset system** — save, load, rename, and delete named configuration profiles.
  - **Config persistence** — all settings auto-saved to `GUI.config` (JSON) on every change.
  - **Test render** button — single-seed preview with sbsrender + GPU path tracer.
  - **Dual HDR test** — renders with Dark.hdr and Bright.hdr side-by-side for exposure validation.
  - **CSV info panel** — displays count of completed seeds.
  - Seed randomiser and manual entry.
  - sbsar library support (main + JSON-based material lists).
  - Seeds-per-material multiplier.
  - Threaded generation with cancel support.
  - Progress status bar and inline image preview.

---

## [0.1.0] — 2026-04-12

### Added
- **Initial dataset generation script** (`generate_pbr_dataset.py`).
  - Automated sbsrender CLI integration for Substance Designer `.sbsar` files.
  - Random 6-digit seed generation for material parameter variation.
  - Dual sbsrender passes: PNG for all channels, HDR specifically for environment maps.
  - `discover_channels()` for detecting exported texture maps with `.hdr` preference for environment.
  - CSV-based dataset tracking with automatic resume on re-run.
  - Auto-exposure from HDR environment map luminance (key=0.6, log-average metering).
  - Glossiness→roughness inversion for materials that export glossiness instead of roughness.
  - Configurable output format (PNG/TGA/EXR/TIF) and resolution (power-of-2).
  - Ctrl-C safe with completed seed skip.
