# PBR Training Dataset Generator

A GPU-accelerated pipeline for generating large-scale PBR (Physically Based Rendering) material datasets. It combines **Substance Designer's sbsrender** CLI for procedural texture baking with a custom OpenGL 4.3 compute-shader path tracer for physically accurate material preview rendering.

![Shaderball Render](neural-network-18m7jp.max-1500x1500.png)

## Features

- **Automated material generation** — Randomises Substance Designer `.sbsar` parameters with unique 6-digit seeds, producing diverse PBR channel maps (basecolor, normal, roughness, metallic, height).
- **GPU path-traced renderballs** — Real-time OpenGL 4.3 compute-shader path tracer with the StandardShaderBall model (24k verts, 46k faces), BVH acceleration, and configurable bounce depth.
- **HDR environment lighting** — Loads RGBE `.hdr` environment maps (from sbsrender output or user-provided HDRIs) with correct float32 radiance decoding via OpenCV.
- **Automatic exposure** — Log-average luminance metering with coverage correction for partially-dark environments, configurable min/max/bias clamps.
- **Multi-angle rendering** — Each material is rendered from multiple configurable camera angles (azimuth + elevation per view).
- **Tkinter GUI** — Full-featured desktop interface with real-time preview, preset management, dual HDR exposure testing, and persistent JSON config.
- **Resumable generation** — CSV-tracked progress; Ctrl-C safe with automatic skip of completed seeds on re-run.
- **HybridPBRDataset** — CSV-driven dataset for model training that collects per-sample PBR maps, renderballs, and conditioning scalars.
- **Refactored PBRNet** — Multi-input neural network accepting image + conditioning scalars for advanced PBR prediction.

## Project Structure

```
PBR_Training/
├── generate_pbr_dataset.py    ← Core dataset generation script
├── pbr_dataset_gui.py         ← Tkinter GUI front-end
├── random-material.sbsar      ← Substance Designer archive (input)
├── GUI.config                 ← Persistent GUI settings (JSON)
├── HDR_Exposure_Test/         ← Test HDRIs (Bright.hdr, Dark.hdr)
├── PBR_Dataset/               ← Output directory
│   ├── dataset.csv            ← Master index of all rendered materials
│   └── renders/
│       └── <seed>/            ← Per-seed folder (e.g. 209510/)
│           ├── <seed>_basecolor.png
│           ├── <seed>_normal.png
│           ├── <seed>_roughness.png
│           ├── <seed>_metallic.png
│           ├── <seed>_height.png
│           ├── <seed>_environment.hdr
│           ├── <seed>_renderball_front.png
│           ├── <seed>_renderball_left.png
│           └── <seed>_renderball_right.png
├── PBR_Training.ipynb         ← Jupyter notebook for model training (HybridPBRDataset + refactored PBRNet)
├── backups/
│   └── PBR_Training.ipynb.bak ← Original notebook backup
└── README.md
```

### Companion Repository

The GPU path tracer lives in a separate repository:

```
PBR-Python-Shaderball-Renderer/
├── python_renderer/
│   ├── core/          ← Material definitions, RenderConfig
│   ├── gpu/
│   │   ├── pathtracer.py      ← GPUPathtracer (OpenGL compute)
│   │   ├── buffer.py          ← GPU buffer/texture wrappers
│   │   └── shaders/
│   │       ├── pathtracer.glsl  ← Main path tracing integrator
│   │       ├── common.glsl      ← RNG, TBN, VNDF sampling, Fresnel
│   │       └── brdf.glsl        ← Smith G1/G2, GGX NDF
│   ├── loaders/
│   │   └── texture_loader.py  ← HDR/texture loading (cv2 primary)
│   └── assets/
│       └── standard-shader-ball/  ← StandardShaderBall .obj/.glb
```

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with OpenGL 4.3+ support (tested on RTX 3060)
- **Substance Designer** (for `sbsrender.exe` CLI)
- **OpenCV** (`pip install opencv-python`) — required for correct HDR/RGBE decoding

### Python Dependencies

```
numpy>=1.21.0
Pillow>=9.0.0
PyOpenGL>=3.1.5
PyGLM>=2.6.0
trimesh>=3.15.0
opencv-python>=4.5.0
glfw>=2.5.0
imageio>=2.9.0
scipy>=1.7.0
lpips>=0.1.4
```

## Quick Start

### Command Line: Dataset Generation with generate_pbr_dataset.py

```bash
python generate_pbr_dataset.py
```

**generate_pbr_dataset.py** is the core dataset generation script that automates the creation of large-scale PBR material datasets:

**Workflow:**
1. Locates `sbsrender.exe` automatically
2. Generates `NUM_MATERIALS` (default 1000) random 6-digit seeds
3. For each seed:
   - Bakes all PBR texture channels via Substance Designer's sbsrender (basecolor, normal, roughness, metallic, height)
   - Renders multi-angle shaderball previews using the GPU path tracer
   - Logs metadata to dataset.csv
4. Creates organized per-seed folders with all outputs
5. Maintains CSV index for easy dataset loading in machine learning frameworks

**Key Features:**
- **Resumable progress tracking** — Ctrl-C safe with automatic skip of completed seeds on re-run
- **CSV-based indexing** — Seed IDs, parameters, and file paths for training
- **Configurable materials** — Material count, resolution, render settings all adjustable
- **Batch processing** — Efficient GPU utilization for large-scale generation
- **Error handling** — Robust handling of sbsrender failures and edge cases

### GUI: Interactive Dataset Generation with pbr_dataset_gui.py

```bash
python pbr_dataset_gui.py
```

**pbr_dataset_gui.py** provides a full-featured Tkinter desktop interface for interactive dataset generation and management:

**Interface Controls:**
- **Material Management**
  - Material count selector (e.g., 100, 1000, 10000)
  - `.sbsar` file browser for Substance Designer archives
  - Output directory configuration
  
- **Render Settings**
  - Resolution (power-of-2 selector: 256, 512, 1024)
  - GPU samples per pixel (64-512)
  - Maximum bounce depth (1-16)
  - Texture tiling multiplier
  - Camera distance and field-of-view (FOV)
  
- **Multi-angle Camera Orbits**
  - Per-view configuration (e.g., "Front", "Left", "Right")
  - Azimuth angle (0-360°) for each view
  - Elevation angle (-90 to +90°) for each view
  - Add/remove custom views
  
- **Auto-Exposure Controls**
  - Min/max exposure clamps
  - Log2 bias adjustment for exposure tuning
  
- **HDR Environment**
  - Override path for custom `.hdr` lighting
  - Environment map preview
  
- **Output Configuration**
  - Toggle individual PBR channels (basecolor, normal, roughness, metallic, height)
  - Compression and quality settings
  
- **Preset Management**
  - Save named configuration profiles
  - Load/rename/delete presets
  - Share presets across projects
  
- **Preview & Testing**
  - **Test Render** — Single-seed preview with sbsrender + path tracer before batch generation
  - **Dual HDR Test** — Side-by-side Dark/Bright HDRI comparison for exposure validation
  - Real-time preview updates  
  
- **Persistent Configuration**
  - All settings saved to `GUI.config` (JSON format)
  - Automatic restoration on next launch
  - Reproducible dataset generation

## Configuration

### generate_pbr_dataset.py Settings

Key settings at the top of `generate_pbr_dataset.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `NUM_MATERIALS` | 1000 | Number of unique materials to generate |
| `OUTPUT_RESOLUTION` | 512 | Texture resolution (power of 2) |
| `SPHERE_RENDER_SIZE` | 512 | Renderball image resolution |
| `GPU_RENDER_SAMPLES` | 256 | Path tracing samples per pixel |
| `GPU_MAX_BOUNCES` | 6 | Maximum light bounce depth |
| `CAMERA_DISTANCE` | 23.4 | Camera distance from shaderball (cm) |
| `CAMERA_FOV` | 23.67° | Vertical field of view |
| `TEXTURE_TILING` | 5.0 | UV tile repeat for PBR textures |
| `AUTO_EXPOSURE_MIN` | 0.005 | Minimum exposure clamp |
| `AUTO_EXPOSURE_MAX` | 50.0 | Maximum exposure clamp |
| `AUTO_EXPOSURE_BIAS` | 0.0 | Exposure bias in log2 stops |

## Training Notebook: PBR_Training.ipynb (Updated)

The training notebook has been refactored to support advanced dataset and model training workflows. Key updates from commit 4b58734:

### HybridPBRDataset

A new CSV-driven PyTorch Dataset class for flexible PBR training:

**Functionality:**
- Loads per-sample PBR maps (basecolor, normal, roughness, metallic, height) from disk
- Loads corresponding renderball images (multiple angles) and conditioning scalars
- Returns tuples: `(input_image, cond_tensor, pbr_targets)`
- Validates `dataset.csv` presence and emits helpful error messages
- Supports flexible conditioning based on material parameters

**Data Structure:**
- Input: Renderball image (256×256 RGB)
- Conditioning: Material scalars (glossiness, texture_scale, etc.)
- Targets: Per-channel PBR maps (256×256)

**Usage Example:**
```python
from PBR_Training import HybridPBRDataset
from torch.utils.data import DataLoader

dataset = HybridPBRDataset(
    csv_path='PBR_Dataset/dataset.csv',
    img_root='PBR_Dataset/renders/'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for input_img, cond_tensor, pbr_targets in dataloader:
    # input_img: (B, 3, 256, 256)
    # cond_tensor: (B, num_scalars)
    # pbr_targets: (B, 5, 256, 256) - for 5 PBR channels
    pass
```

**Dataset Generation:**
If `dataset.csv` is missing, first run:
```bash
python generate_pbr_dataset.py
```

### Refactored PBRNet

The neural network architecture has been refactored to accept multi-modal inputs and produce high-quality PBR texture predictions:

**Architecture Overview:**
- **Image Encoder**: Processes input renderball (256×256 RGB) → feature map
- **Conditioning Encoder**: Encodes material parameter scalars → condition vector
- **Fusion Layer**: Combines image and conditioning features via concatenation/gating
- **Texture Decoder**: Upsamples fused features → 256×256 PBR channel predictions
- **Scalar Prediction Head**: Predicts material conditioning parameters
- **Skip Connections**: Preserve fine details through decoder path

**Parameter Registry Updates:**
Standardized PBR parameter keys for consistency:
- `basecolor` — RGB diffuse color
- `normal` — Normal map in tangent space
- `roughness` — Microfacet roughness [0-1]
- `metallic` — Metallicity [0-1]
- `height` — Height/displacement map
- `glossiness` — Now enabled by default (inverse of roughness)

**Key Improvements:**
- Texture resolution: 256×256 (up from prior versions)
- Decoder output size: 256 for finer detail preservation
- LPIPS (Learned Perceptual Image Patch Similarity) loss integrated for perceptually-guided training
- Conditioning-aware predictions improve material consistency

**Training Loop Example:**
```python
from PBR_Training import HybridPBRDataset, PBRNet
import torch
import torch.nn as nn

model = PBRNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion_mse = nn.MSELoss()

dataset = HybridPBRDataset(...)
dataloader = DataLoader(dataset, batch_size=32)

for epoch in range(num_epochs):
    for input_img, cond, targets in dataloader:
        pred_maps, pred_cond = model(input_img, cond)
        
        # Combine MSE and LPIPS losses
        loss_mse = criterion_mse(pred_maps, targets)
        loss_perceptual = lpips_loss(pred_maps, targets)
        loss = loss_mse + 0.1 * loss_perceptual
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Notebook Features

- **Environment Setup** — Install line includes all dependencies: numpy, torch, lpips, cv2, etc.
- **CSV-driven Workflows** — Automatic dataset discovery and loading from `dataset.csv`
- **Data Visualization** — Training progress monitoring with loss curves and rendered sample comparisons
- **Model Checkpointing** — Save/load best models during training
- **Hyperparameter Tuning** — Easy adjustment of:
  - Learning rates and schedulers
  - Batch sizes and data augmentation
  - Model architecture (encoder depth, fusion mechanism)
  - Loss function weights (MSE, LPIPS, regularization)
- **Inference Pipeline** — Generate PBR maps from new renderball images

## Technical Details

### Path Tracer

- OpenGL 4.3 compute shader running headless via GLFW
- ACES filmic tonemapping with configurable exposure
- VNDF (Visible Normal Distribution Function) importance sampling (Dupuy & Benyoub 2023)
- Smith-GGX specular BRDF with correct denominator `4·NdotL·NdotV`
- NaN/Inf guards at accumulation and throughput levels
- BVH-accelerated ray-mesh intersection

### Auto-Exposure

The auto-exposure system computes a log-average luminance from the HDR environment map, then derives exposure as:

```
exposure = (key / log_avg_luminance) × 2^bias
```

Where `key = 0.6`. A **coverage correction** boosts exposure when less than 100% of the environment provides significant light (e.g., a dark HDRI with a small bright region), preventing underexposure in unevenly-lit scenes.

### HDR Loading

Environment maps are loaded via **OpenCV** (`cv2.imread` with `IMREAD_UNCHANGED`) which correctly decodes RGBE `.hdr` files to float32 radiance values. This replaces an earlier `imageio`-based approach for better performance and correctness.

## License

See individual repository licenses.