"""
pbr_dataset_gui.py
──────────────────────────────────────────────────────────────────────────────
GUI front-end for the PBR Dataset Generator (generate_pbr_dataset.py).
Requires: Python 3.10+, tkinter (bundled with Python).
"""

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

# ── Ensure the PBR renderer is on the path ───────────────────────────────────
PBR_RENDERER_PATH = r"F:\!_GitHub_Rep\PBR-Python-Shaderball-Renderer"
if PBR_RENDERER_PATH not in sys.path:
    sys.path.insert(0, PBR_RENDERER_PATH)

# ── Import the dataset generator module ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pbr_dataset as gen

# ── Constants ─────────────────────────────────────────────────────────────────
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_render_cache")
SBSAR_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbsar_library.json")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUI.config")

ALL_CHANNELS = [
    "basecolor", "normal", "roughness", "metallic", "height",
    "ambientocclusion", "specular", "glossiness", "background", "environment",
]

FORMAT_OPTIONS = ["png", "tga", "exr", "tif"]
RESOLUTION_OPTIONS = [128, 256, 512, 1024, 2048, 4096]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIG PERSISTENCE                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _default_settings() -> dict:
    """Return a settings dict with current generator defaults."""
    return {
        "num_materials":      gen.NUM_MATERIALS,
        "sbsar_path":         gen.SBSAR_PATH,
        "output_root":        gen.OUTPUT_ROOT,
        "output_format":      gen.OUTPUT_FORMAT,
        "output_resolution":  gen.OUTPUT_RESOLUTION,
        "sphere_render_size": gen.SPHERE_RENDER_SIZE,
        "displacement_scale": gen.DISPLACEMENT_SCALE,
        "hdr_environment":    gen.HDR_ENVIRONMENT_PATH,
        "gpu_samples":        gen.GPU_RENDER_SAMPLES,
        "max_bounces":        gen.GPU_MAX_BOUNCES,
        "texture_tiling":     gen.TEXTURE_TILING,
        "camera_distance":    gen.CAMERA_DISTANCE,
        "camera_fov":         gen.CAMERA_FOV,
        "exposure_min":       gen.AUTO_EXPOSURE_MIN,
        "exposure_max":       gen.AUTO_EXPOSURE_MAX,
        "exposure_bias":      gen.AUTO_EXPOSURE_BIAS,
        "channels":           list(gen.CHANNEL_NAMES),
        "angles":             list(gen.RENDERBALL_ANGLES),
        "use_main_sbsar":     True,
        "use_json_sbsar":     False,
        "seeds_per_material": 1,
    }


def _load_config() -> dict:
    """Load the GUI.config file, returning the full config structure."""
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_preset": "Default", "presets": {"Default": _default_settings()}}


def _save_config(config: dict):
    """Write config to GUI.config."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN GUI                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PBRDatasetGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PBR Dataset Generator")
        self.root.minsize(820, 900)

        self._suppress_save = True  # prevent saves during init

        # Load config
        self._config = _load_config()
        self._active_preset = self._config.get("active_preset", "Default")
        if self._active_preset not in self._config.get("presets", {}):
            self._active_preset = "Default"
            self._config.setdefault("presets", {}).setdefault("Default", _default_settings())

        self._build_vars()
        self._apply_preset(self._active_preset)
        self._build_ui()
        self._setup_auto_save()
        self._refresh_csv_info()

        self._suppress_save = False

    # ──────────────────────────────────────────────────────────────────────
    #  Variables
    # ──────────────────────────────────────────────────────────────────────
    def _build_vars(self):
        self.v_num_materials    = tk.IntVar(value=gen.NUM_MATERIALS)
        self.v_sbsar_path       = tk.StringVar(value=gen.SBSAR_PATH)
        self.v_output_root      = tk.StringVar(value=gen.OUTPUT_ROOT)
        self.v_output_format    = tk.StringVar(value=gen.OUTPUT_FORMAT)
        self.v_output_res       = tk.IntVar(value=gen.OUTPUT_RESOLUTION)
        self.v_sphere_size      = tk.IntVar(value=gen.SPHERE_RENDER_SIZE)
        self.v_disp_scale       = tk.DoubleVar(value=gen.DISPLACEMENT_SCALE)
        self.v_hdr_env          = tk.StringVar(value=gen.HDR_ENVIRONMENT_PATH)
        self.v_gpu_samples      = tk.IntVar(value=gen.GPU_RENDER_SAMPLES)
        self.v_max_bounces      = tk.IntVar(value=gen.GPU_MAX_BOUNCES)
        self.v_tex_tiling       = tk.DoubleVar(value=gen.TEXTURE_TILING)
        self.v_camera_dist      = tk.DoubleVar(value=gen.CAMERA_DISTANCE)
        self.v_camera_fov       = tk.DoubleVar(value=gen.CAMERA_FOV)

        self.v_exp_min          = tk.DoubleVar(value=gen.AUTO_EXPOSURE_MIN)
        self.v_exp_max          = tk.DoubleVar(value=gen.AUTO_EXPOSURE_MAX)
        self.v_exp_bias         = tk.DoubleVar(value=gen.AUTO_EXPOSURE_BIAS)

        # Test seed
        self.v_test_seed        = tk.StringVar(value="")

        # Preset selector
        self.v_preset           = tk.StringVar(value=self._active_preset)

        # Renderball angles (list of dicts)
        self.angle_rows: list[dict] = []

        # Channel checkboxes
        self.channel_vars: dict[str, tk.BooleanVar] = {}
        for ch in ALL_CHANNELS:
            self.channel_vars[ch] = tk.BooleanVar(value=(ch in gen.CHANNEL_NAMES))

        # sbsar library
        self.v_use_main_sbsar   = tk.BooleanVar(value=True)
        self.v_use_json_sbsar   = tk.BooleanVar(value=False)
        self.v_seeds_per_mat    = tk.IntVar(value=1)

        # Running state
        self._running = False
        self._cancel = False

    # ──────────────────────────────────────────────────────────────────────
    #  Preset / Config helpers
    # ──────────────────────────────────────────────────────────────────────
    def _settings_to_dict(self) -> dict:
        """Read current GUI state into a dict."""
        angles = []
        for r in self.angle_rows:
            lbl = r["label"].get().strip()
            if lbl:
                angles.append({"label": lbl, "azimuth_deg": r["azimuth"].get(), "elevation_deg": r["elevation"].get()})
        channels = [ch for ch in ALL_CHANNELS if self.channel_vars[ch].get()]

        return {
            "num_materials":      self.v_num_materials.get(),
            "sbsar_path":         self.v_sbsar_path.get(),
            "output_root":        self.v_output_root.get(),
            "output_format":      self.v_output_format.get(),
            "output_resolution":  int(self.v_output_res.get()),
            "sphere_render_size": self.v_sphere_size.get(),
            "displacement_scale": self.v_disp_scale.get(),
            "hdr_environment":    self.v_hdr_env.get(),
            "gpu_samples":        self.v_gpu_samples.get(),
            "max_bounces":        self.v_max_bounces.get(),
            "texture_tiling":     self.v_tex_tiling.get(),
            "camera_distance":    self.v_camera_dist.get(),
            "camera_fov":         self.v_camera_fov.get(),
            "exposure_min":       self.v_exp_min.get(),
            "exposure_max":       self.v_exp_max.get(),
            "exposure_bias":      self.v_exp_bias.get(),
            "channels":           channels,
            "angles":             angles,
            "use_main_sbsar":     self.v_use_main_sbsar.get(),
            "use_json_sbsar":     self.v_use_json_sbsar.get(),
            "seeds_per_material": self.v_seeds_per_mat.get(),
        }

    def _dict_to_vars(self, d: dict):
        """Apply a settings dict to the GUI variables."""
        self._suppress_save = True
        try:
            self.v_num_materials.set(d.get("num_materials", gen.NUM_MATERIALS))
            self.v_sbsar_path.set(d.get("sbsar_path", gen.SBSAR_PATH))
            self.v_output_root.set(d.get("output_root", gen.OUTPUT_ROOT))
            self.v_output_format.set(d.get("output_format", gen.OUTPUT_FORMAT))
            self.v_output_res.set(d.get("output_resolution", gen.OUTPUT_RESOLUTION))
            self.v_sphere_size.set(d.get("sphere_render_size", gen.SPHERE_RENDER_SIZE))
            self.v_disp_scale.set(d.get("displacement_scale", gen.DISPLACEMENT_SCALE))
            self.v_hdr_env.set(d.get("hdr_environment", gen.HDR_ENVIRONMENT_PATH))
            self.v_gpu_samples.set(d.get("gpu_samples", gen.GPU_RENDER_SAMPLES))
            self.v_max_bounces.set(d.get("max_bounces", gen.GPU_MAX_BOUNCES))
            self.v_tex_tiling.set(d.get("texture_tiling", gen.TEXTURE_TILING))
            self.v_camera_dist.set(d.get("camera_distance", gen.CAMERA_DISTANCE))
            self.v_camera_fov.set(d.get("camera_fov", gen.CAMERA_FOV))
            self.v_exp_min.set(d.get("exposure_min", gen.AUTO_EXPOSURE_MIN))
            self.v_exp_max.set(d.get("exposure_max", gen.AUTO_EXPOSURE_MAX))
            self.v_exp_bias.set(d.get("exposure_bias", gen.AUTO_EXPOSURE_BIAS))
            self.v_use_main_sbsar.set(d.get("use_main_sbsar", True))
            self.v_use_json_sbsar.set(d.get("use_json_sbsar", False))
            self.v_seeds_per_mat.set(d.get("seeds_per_material", 1))

            # Channels
            ch_list = d.get("channels", list(gen.CHANNEL_NAMES))
            for ch in ALL_CHANNELS:
                self.channel_vars[ch].set(ch in ch_list)

            # Angles — rebuild rows if angles_frame exists
            if hasattr(self, "angles_frame"):
                for r in list(self.angle_rows):
                    r["frame"].destroy()
                self.angle_rows.clear()
                for a in d.get("angles", gen.RENDERBALL_ANGLES):
                    self._add_angle(a.get("label", ""), a.get("azimuth_deg", 0), a.get("elevation_deg", 38))
        finally:
            self._suppress_save = False

    def _apply_preset(self, name: str):
        """Load a preset by name into the GUI vars."""
        presets = self._config.get("presets", {})
        d = presets.get(name, _default_settings())
        self._dict_to_vars(d)
        self._active_preset = name
        self._config["active_preset"] = name

    def _auto_save(self, *_args):
        """Called by variable traces — saves current state to active preset."""
        if self._suppress_save:
            return
        try:
            d = self._settings_to_dict()
            self._config.setdefault("presets", {})[self._active_preset] = d
            self._config["active_preset"] = self._active_preset
            _save_config(self._config)
        except (tk.TclError, ValueError):
            pass  # ignore partial edits (e.g. empty entry field)

    def _setup_auto_save(self):
        """Attach traces to all settings variables for auto-save."""
        tracked_vars = [
            self.v_num_materials, self.v_sbsar_path, self.v_output_root,
            self.v_output_format, self.v_output_res, self.v_sphere_size,
            self.v_disp_scale, self.v_hdr_env, self.v_gpu_samples,
            self.v_max_bounces, self.v_tex_tiling, self.v_camera_dist,
            self.v_camera_fov, self.v_exp_min, self.v_exp_max, self.v_exp_bias,
            self.v_use_main_sbsar, self.v_use_json_sbsar, self.v_seeds_per_mat,
        ]
        for v in tracked_vars:
            v.trace_add("write", self._auto_save)
        for ch_var in self.channel_vars.values():
            ch_var.trace_add("write", self._auto_save)

    # ──────────────────────────────────────────────────────────────────────
    #  UI Layout
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        parent = self.scroll_frame
        pad = {"padx": 6, "pady": 3}

        # ── Section: Presets ──────────────────────────────────────────────
        sec_preset = ttk.LabelFrame(parent, text="Presets", padding=8)
        sec_preset.pack(fill="x", **pad)

        preset_row = ttk.Frame(sec_preset)
        preset_row.pack(fill="x")

        ttk.Label(preset_row, text="Active Preset:").pack(side="left")
        preset_names = list(self._config.get("presets", {}).keys())
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.v_preset,
            values=preset_names, width=20, state="readonly")
        self.preset_combo.pack(side="left", padx=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Button(preset_row, text="Save As New...",
                   command=self._save_preset_as).pack(side="left", padx=4)
        ttk.Button(preset_row, text="Delete Preset",
                   command=self._delete_preset).pack(side="left", padx=4)

        # ── Section: General Settings ─────────────────────────────────────
        sec = ttk.LabelFrame(parent, text="General Settings", padding=8)
        sec.pack(fill="x", **pad)

        self._row_entry(sec, "Num Materials:", self.v_num_materials, 0)
        self._row_browse(sec, "SBSAR Path:", self.v_sbsar_path, 1,
                         filetypes=[("Substance Archive", "*.sbsar")])
        self._row_browse(sec, "Output Root:", self.v_output_root, 2, directory=True)
        self._row_combo(sec, "Output Format:", self.v_output_format, FORMAT_OPTIONS, 3)
        self._row_combo(sec, "Output Resolution:", self.v_output_res,
                        RESOLUTION_OPTIONS, 4)

        # ── Section: Render Settings ──────────────────────────────────────
        sec2 = ttk.LabelFrame(parent, text="Render Settings", padding=8)
        sec2.pack(fill="x", **pad)

        self._row_entry(sec2, "Sphere Render Size:", self.v_sphere_size, 0)
        self._row_entry(sec2, "Displacement Scale:", self.v_disp_scale, 1)
        self._row_browse(sec2, "HDR Environment:", self.v_hdr_env, 2,
                         filetypes=[("HDR/EXR", "*.hdr *.exr"), ("All", "*.*")])
        self._row_entry(sec2, "GPU Samples:", self.v_gpu_samples, 3)
        self._row_entry(sec2, "Max Bounces:", self.v_max_bounces, 4)
        self._row_entry(sec2, "Texture Tiling:", self.v_tex_tiling, 5)
        self._row_entry(sec2, "Camera Distance:", self.v_camera_dist, 6)
        self._row_entry(sec2, "Camera FOV\u00b0:", self.v_camera_fov, 7)

        # ── Section: Auto-Exposure ────────────────────────────────────────
        sec_exp = ttk.LabelFrame(parent, text="Auto-Exposure", padding=8)
        sec_exp.pack(fill="x", **pad)

        ttk.Label(sec_exp, text="Min Exposure:").grid(row=0, column=0, sticky="w")
        ttk.Entry(sec_exp, textvariable=self.v_exp_min, width=10).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(sec_exp, text="Max Exposure:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(sec_exp, textvariable=self.v_exp_max, width=10).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(sec_exp, text="Bias (log2 stops):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        bias_frame = ttk.Frame(sec_exp)
        bias_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        self.bias_slider = ttk.Scale(bias_frame, from_=-4.0, to=4.0,
                                     variable=self.v_exp_bias, orient="horizontal")
        self.bias_slider.pack(side="left", fill="x", expand=True)
        self.bias_label = ttk.Label(bias_frame, text=f"{self.v_exp_bias.get():.2f}")
        self.bias_label.pack(side="left", padx=4)
        self.v_exp_bias.trace_add("write", lambda *_: self.bias_label.config(
            text=f"{self.v_exp_bias.get():.2f}"))

        # ── Section: Renderball Angles ────────────────────────────────────
        sec_ang = ttk.LabelFrame(parent, text="Renderball Angles", padding=8)
        sec_ang.pack(fill="x", **pad)

        self.angles_frame = ttk.Frame(sec_ang)
        self.angles_frame.pack(fill="x")

        btn_row = ttk.Frame(sec_ang)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="+ Add Angle", command=self._add_angle).pack(side="left")

        # Populate angles from the active preset
        if not self.angle_rows:
            preset_data = self._config.get("presets", {}).get(
                self._active_preset, _default_settings())
            for a in preset_data.get("angles", gen.RENDERBALL_ANGLES):
                self._add_angle(a.get("label", ""), a.get("azimuth_deg", 0), a.get("elevation_deg", 38))

        # ── Section: Channels ─────────────────────────────────────────────
        sec_ch = ttk.LabelFrame(parent, text="Output Channels (checkboxes)", padding=8)
        sec_ch.pack(fill="x", **pad)

        ch_frame = ttk.Frame(sec_ch)
        ch_frame.pack(fill="x")
        for i, ch in enumerate(ALL_CHANNELS):
            ttk.Checkbutton(ch_frame, text=ch, variable=self.channel_vars[ch]).grid(
                row=i // 5, column=i % 5, sticky="w", padx=4, pady=2)

        # ── Section: CSV / Dataset Info ───────────────────────────────────
        sec_csv = ttk.LabelFrame(parent, text="Dataset Info", padding=8)
        sec_csv.pack(fill="x", **pad)

        self.csv_info_label = ttk.Label(sec_csv, text="No CSV found.")
        self.csv_info_label.pack(anchor="w")

        self.total_label = ttk.Label(sec_csv, text="", font=("", 9, "bold"))
        self.total_label.pack(anchor="w", pady=(4, 0))

        ttk.Button(sec_csv, text="Refresh", command=self._refresh_csv_info).pack(
            anchor="w", pady=(4, 0))

        # ── Section: SBSAR Library ────────────────────────────────────────
        sec_lib = ttk.LabelFrame(parent, text="SBSAR Material Library", padding=8)
        sec_lib.pack(fill="x", **pad)

        chk_row = ttk.Frame(sec_lib)
        chk_row.pack(fill="x")
        ttk.Checkbutton(chk_row, text="Use main SBSAR", variable=self.v_use_main_sbsar).pack(side="left")
        ttk.Checkbutton(chk_row, text="Use JSON library", variable=self.v_use_json_sbsar).pack(side="left", padx=(12, 0))

        seed_row = ttk.Frame(sec_lib)
        seed_row.pack(fill="x", pady=(4, 0))
        ttk.Label(seed_row, text="Seeds per material:").pack(side="left")
        ttk.Entry(seed_row, textvariable=self.v_seeds_per_mat, width=6).pack(side="left", padx=4)

        self.v_seeds_per_mat.trace_add("write", lambda *_: self._update_total_label())
        self.v_num_materials.trace_add("write", lambda *_: self._update_total_label())
        self.v_use_main_sbsar.trace_add("write", lambda *_: self._update_total_label())
        self.v_use_json_sbsar.trace_add("write", lambda *_: self._update_total_label())

        search_row = ttk.Frame(sec_lib)
        search_row.pack(fill="x", pady=(4, 0))
        ttk.Button(search_row, text="Search for .sbsar files...",
                   command=self._search_sbsar).pack(side="left")
        self.sbsar_lib_label = ttk.Label(search_row, text="")
        self.sbsar_lib_label.pack(side="left", padx=8)
        self._update_lib_label()

        self.search_progress = ttk.Progressbar(sec_lib, mode="determinate")
        self.search_progress.pack(fill="x", pady=(4, 0))
        self.search_status = ttk.Label(sec_lib, text="")
        self.search_status.pack(anchor="w")

        # ── Section: Preview & Actions ────────────────────────────────────
        sec_act = ttk.LabelFrame(parent, text="Preview & Actions", padding=8)
        sec_act.pack(fill="x", **pad)

        # Seed entry + Randomize
        seed_frame = ttk.Frame(sec_act)
        seed_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(seed_frame, text="Test Seed:").pack(side="left")
        ttk.Entry(seed_frame, textvariable=self.v_test_seed, width=12).pack(side="left", padx=4)
        ttk.Button(seed_frame, text="Randomize",
                   command=self._randomize_seed).pack(side="left", padx=4)

        # Action buttons
        btn_row2 = ttk.Frame(sec_act)
        btn_row2.pack(fill="x")

        ttk.Button(btn_row2, text="Render Test",
                   command=self._render_test).pack(side="left", padx=4)
        ttk.Button(btn_row2, text="Dual HDR Test",
                   command=self._dual_hdr_test).pack(side="left", padx=4)
        ttk.Button(btn_row2, text="Open Gallery",
                   command=self._open_gallery).pack(side="left", padx=4)

        # Preview area — frame that holds 1 or 2 images side-by-side
        self.preview_frame = ttk.Frame(sec_act)
        self.preview_frame.pack(pady=8)
        self._preview_placeholder = ttk.Label(self.preview_frame, text="(no preview)")
        self._preview_placeholder.pack()
        self._thumb_photos = []  # prevent GC

        # ── Section: Generate ─────────────────────────────────────────────
        sec_gen = ttk.LabelFrame(parent, text="Generate Dataset", padding=8)
        sec_gen.pack(fill="x", **pad)

        self.gen_progress = ttk.Progressbar(sec_gen, mode="determinate")
        self.gen_progress.pack(fill="x")
        self.gen_status = ttk.Label(sec_gen, text="Ready.")
        self.gen_status.pack(anchor="w", pady=(4, 0))

        gen_btn_row = ttk.Frame(sec_gen)
        gen_btn_row.pack(fill="x", pady=(4, 0))
        self.btn_generate = ttk.Button(gen_btn_row, text="Create Dataset",
                                       command=self._generate_dataset)
        self.btn_generate.pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(gen_btn_row, text="Cancel", state="disabled",
                                     command=self._cancel_generation)
        self.btn_cancel.pack(side="left", padx=4)

        self._update_total_label()

    # ──────────────────────────────────────────────────────────────────────
    #  UI Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _row_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        ttk.Entry(parent, textvariable=var, width=40).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

    def _row_browse(self, parent, label, var, row, filetypes=None, directory=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        ttk.Entry(parent, textvariable=var, width=40).grid(
            row=row, column=1, sticky="ew", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

        def _pick():
            if directory:
                p = filedialog.askdirectory(initialdir=var.get() or ".")
            else:
                p = filedialog.askopenfilename(
                    initialdir=os.path.dirname(var.get()) or ".",
                    filetypes=filetypes or [("All", "*.*")])
            if p:
                var.set(p)
        ttk.Button(parent, text="Browse...", command=_pick).grid(
            row=row, column=2, padx=4, pady=2)

    def _row_combo(self, parent, label, var, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        cb = ttk.Combobox(parent, textvariable=var,
                          values=[str(v) for v in values], width=12)
        cb.grid(row=row, column=1, sticky="w", padx=4, pady=2)

    def _randomize_seed(self):
        self.v_test_seed.set(gen.generate_seed())

    # ── Angle rows ────────────────────────────────────────────────────────
    def _add_angle(self, label_text="", azimuth_val=0, elevation_val=38):
        row_frame = ttk.Frame(self.angles_frame)
        row_frame.pack(fill="x", pady=1)

        lbl_var = tk.StringVar(value=label_text)
        az_var  = tk.DoubleVar(value=azimuth_val)
        el_var  = tk.DoubleVar(value=elevation_val)

        ttk.Label(row_frame, text="Label:").pack(side="left")
        ttk.Entry(row_frame, textvariable=lbl_var, width=10).pack(side="left", padx=2)
        ttk.Label(row_frame, text="Azimuth\u00b0:").pack(side="left", padx=(8, 0))
        ttk.Entry(row_frame, textvariable=az_var, width=8).pack(side="left", padx=2)
        ttk.Label(row_frame, text="Elevation\u00b0:").pack(side="left", padx=(8, 0))
        ttk.Entry(row_frame, textvariable=el_var, width=8).pack(side="left", padx=2)

        def _remove():
            self.angle_rows = [r for r in self.angle_rows if r["frame"] is not row_frame]
            row_frame.destroy()
            self._auto_save()
        ttk.Button(row_frame, text="\u00d7", width=2, command=_remove).pack(side="left", padx=4)

        self.angle_rows.append({"frame": row_frame, "label": lbl_var, "azimuth": az_var, "elevation": el_var})

    # ──────────────────────────────────────────────────────────────────────
    #  Preset management
    # ──────────────────────────────────────────────────────────────────────
    def _on_preset_selected(self, _event=None):
        name = self.v_preset.get()
        if name and name != self._active_preset:
            self._apply_preset(name)
            _save_config(self._config)

    def _save_preset_as(self):
        name = simpledialog.askstring("Save Preset", "Enter a name for the new preset:",
                                      parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        self._config.setdefault("presets", {})[name] = self._settings_to_dict()
        self._active_preset = name
        self._config["active_preset"] = name
        self.v_preset.set(name)
        self.preset_combo["values"] = list(self._config["presets"].keys())
        _save_config(self._config)

    def _delete_preset(self):
        name = self.v_preset.get()
        if name == "Default":
            messagebox.showwarning("Delete Preset", "Cannot delete the Default preset.")
            return
        presets = self._config.get("presets", {})
        if name not in presets:
            return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            return
        del presets[name]
        self._active_preset = "Default"
        self._config["active_preset"] = "Default"
        self.v_preset.set("Default")
        self.preset_combo["values"] = list(presets.keys())
        self._apply_preset("Default")
        _save_config(self._config)

    # ──────────────────────────────────────────────────────────────────────
    #  Apply GUI values back into the generator module
    # ──────────────────────────────────────────────────────────────────────
    def _apply_settings(self):
        gen.NUM_MATERIALS       = self.v_num_materials.get()
        gen.SBSAR_PATH          = self.v_sbsar_path.get()
        gen.OUTPUT_ROOT         = self.v_output_root.get()
        gen.OUTPUT_FORMAT       = self.v_output_format.get()
        gen.OUTPUT_RESOLUTION   = int(self.v_output_res.get())
        gen.SPHERE_RENDER_SIZE  = self.v_sphere_size.get()
        gen.DISPLACEMENT_SCALE  = self.v_disp_scale.get()
        gen.HDR_ENVIRONMENT_PATH = self.v_hdr_env.get()
        gen.GPU_RENDER_SAMPLES  = self.v_gpu_samples.get()
        gen.GPU_MAX_BOUNCES     = self.v_max_bounces.get()
        gen.TEXTURE_TILING      = self.v_tex_tiling.get()
        gen.CAMERA_DISTANCE     = self.v_camera_dist.get()
        gen.CAMERA_FOV          = self.v_camera_fov.get()
        gen.AUTO_EXPOSURE_MIN   = self.v_exp_min.get()
        gen.AUTO_EXPOSURE_MAX   = self.v_exp_max.get()
        gen.AUTO_EXPOSURE_BIAS  = self.v_exp_bias.get()

        # Channels
        gen.CHANNEL_NAMES = [ch for ch in ALL_CHANNELS if self.channel_vars[ch].get()]

        # Angles
        gen.RENDERBALL_ANGLES = []
        for r in self.angle_rows:
            lbl = r["label"].get().strip()
            if lbl:
                gen.RENDERBALL_ANGLES.append(
                    {"label": lbl, "azimuth_deg": r["azimuth"].get(), "elevation_deg": r["elevation"].get()})

        # CSV columns depend on channels + angles
        gen.CSV_COLUMNS = ["seed", "folder_path"] + gen.CHANNEL_NAMES + \
                          [f"renderball_{a['label']}" for a in gen.RENDERBALL_ANGLES]

        # Force re-init GPU renderer with new settings
        gen._gpu_pt = None

    # ──────────────────────────────────────────────────────────────────────
    #  CSV / Dataset info
    # ──────────────────────────────────────────────────────────────────────
    def _refresh_csv_info(self):
        csv_path = Path(self.v_output_root.get()) / "dataset.csv"
        if csv_path.is_file():
            rows, seeds = gen.load_existing_csv(csv_path)
            self.csv_info_label.config(
                text=f"CSV: {len(seeds)} seeds stored  \u2022  {csv_path}")
        else:
            self.csv_info_label.config(text="No CSV found.")
        self._update_total_label()

    def _get_json_material_count(self) -> int:
        if os.path.isfile(SBSAR_JSON_PATH):
            try:
                with open(SBSAR_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return len(data.get("materials", []))
            except Exception:
                pass
        return 0

    def _update_total_label(self):
        try:
            seeds_per = self.v_seeds_per_mat.get()
        except (tk.TclError, ValueError):
            seeds_per = 1
        total = 0
        if self.v_use_main_sbsar.get():
            try:
                total += self.v_num_materials.get() * seeds_per
            except (tk.TclError, ValueError):
                pass
        if self.v_use_json_sbsar.get():
            total += self._get_json_material_count() * seeds_per
        self.total_label.config(text=f"Total renders to generate: {total}")

    def _update_lib_label(self):
        count = self._get_json_material_count()
        self.sbsar_lib_label.config(
            text=f"{count} materials in library" if count else "No library loaded")

    # ──────────────────────────────────────────────────────────────────────
    #  Preview helpers
    # ──────────────────────────────────────────────────────────────────────
    def _clear_preview(self):
        for w in self.preview_frame.winfo_children():
            w.destroy()
        self._thumb_photos.clear()

    def _show_preview_images(self, images: list[tuple[str, str]]):
        """Show one or more (label, path) images side-by-side in the preview area."""
        self._clear_preview()
        for label_text, img_path in images:
            cell = ttk.Frame(self.preview_frame)
            cell.pack(side="left", padx=6)
            ttk.Label(cell, text=label_text, font=("", 9, "bold")).pack()
            try:
                img = Image.open(img_path)
                img.thumbnail((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_photos.append(photo)
                ttk.Label(cell, image=photo).pack()
            except Exception as e:
                ttk.Label(cell, text=f"Error: {e}").pack()

    def _get_or_generate_seed(self) -> str:
        """Return the seed from the entry field, or generate a new one."""
        seed = self.v_test_seed.get().strip()
        if not seed:
            seed = gen.generate_seed()
            self.v_test_seed.set(seed)
        return seed

    # ──────────────────────────────────────────────────────────────────────
    #  Render Test
    # ──────────────────────────────────────────────────────────────────────
    def _render_test(self):
        if self._running:
            return
        self._apply_settings()
        self._running = True
        seed = self._get_or_generate_seed()
        self.gen_status.config(text=f"Rendering test preview (seed {seed})...")
        self.root.update()

        def _do_render():
            try:
                os.makedirs(TEMP_DIR, exist_ok=True)
                sbsrender_exe = gen.find_sbsrender()
                if not sbsrender_exe:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "sbsrender not found"))
                    return

                temp_out = Path(TEMP_DIR) / seed
                temp_out.mkdir(parents=True, exist_ok=True)

                # Temporarily redirect output root
                orig_root = gen.OUTPUT_ROOT
                gen.OUTPUT_ROOT = TEMP_DIR

                ok = gen.render_material(sbsrender_exe, seed)
                if not ok:
                    gen.OUTPUT_ROOT = orig_root
                    self.root.after(0, lambda: self.gen_status.config(
                        text="Test render failed (sbsrender)."))
                    return

                channels = gen.discover_channels(seed)
                img_path = gen.render_sphere_preview(
                    channels, seed, azimuth_deg=25.0, suffix="test")

                gen.OUTPUT_ROOT = orig_root
                self.root.after(0, lambda p=img_path: self._show_preview_images(
                    [("Render Test", p)]))
                self.root.after(0, lambda: self.gen_status.config(
                    text=f"Test render complete (seed {seed})."))
            except Exception as e:
                self.root.after(0, lambda: self.gen_status.config(text=f"Error: {e}"))
            finally:
                self._running = False

        threading.Thread(target=_do_render, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    #  Dual HDR Test Render
    # ──────────────────────────────────────────────────────────────────────
    def _dual_hdr_test(self):
        if self._running:
            return

        hdr_dir = r"F:\!_GitHub_Rep\PBR_Training\HDR_Exposure_Test"
        bright_hdr = os.path.join(hdr_dir, "Bright.hdr")
        dark_hdr = os.path.join(hdr_dir, "Dark.hdr")

        if not os.path.isfile(bright_hdr) or not os.path.isfile(dark_hdr):
            messagebox.showerror(
                "Error",
                f"HDR test files not found in:\n{hdr_dir}\n"
                "Expected: Bright.hdr and Dark.hdr")
            return

        self._apply_settings()
        self._running = True
        seed = self._get_or_generate_seed()
        self.gen_status.config(text=f"Rendering dual HDR test (seed {seed})...")
        self.root.update()

        def _do_dual():
            try:
                os.makedirs(TEMP_DIR, exist_ok=True)
                sbsrender_exe = gen.find_sbsrender()
                if not sbsrender_exe:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "sbsrender not found"))
                    return

                temp_out = Path(TEMP_DIR) / seed
                temp_out.mkdir(parents=True, exist_ok=True)

                # Temporarily redirect output root
                orig_root = gen.OUTPUT_ROOT
                gen.OUTPUT_ROOT = TEMP_DIR

                ok = gen.render_material(sbsrender_exe, seed)
                if not ok:
                    gen.OUTPUT_ROOT = orig_root
                    self.root.after(0, lambda: self.gen_status.config(
                        text="Dual HDR test failed (sbsrender)."))
                    return

                channels = gen.discover_channels(seed)

                # Render with Dark.hdr first
                self.root.after(0, lambda: self.gen_status.config(
                    text="Rendering with Dark.hdr..."))
                img_dark = gen.render_sphere_preview(
                    channels, seed, azimuth_deg=25.0, suffix="dark",
                    env_override=dark_hdr)

                # Render with Bright.hdr second
                self.root.after(0, lambda: self.gen_status.config(
                    text="Rendering with Bright.hdr..."))
                img_bright = gen.render_sphere_preview(
                    channels, seed, azimuth_deg=25.0, suffix="bright",
                    env_override=bright_hdr)

                gen.OUTPUT_ROOT = orig_root

                # Show both inline: Dark first, then Bright
                self.root.after(0, lambda d=img_dark, b=img_bright:
                                self._show_preview_images([
                                    ("Dark.hdr", d), ("Bright.hdr", b)]))
                self.root.after(0, lambda: self.gen_status.config(
                    text=f"Dual HDR test complete (seed {seed})."))
            except Exception as e:
                self.root.after(0, lambda: self.gen_status.config(text=f"Error: {e}"))
            finally:
                self._running = False

        threading.Thread(target=_do_dual, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    #  Gallery
    # ──────────────────────────────────────────────────────────────────────
    def _open_gallery(self):
        renders_dir = Path(self.v_output_root.get()) / "renders"
        if not renders_dir.is_dir():
            messagebox.showinfo("Gallery", "No renders directory found.")
            return

        gallery = tk.Toplevel(self.root)
        gallery.title("Shaderball Gallery")
        gallery.geometry("900x700")

        canvas = tk.Canvas(gallery)
        vscroll = ttk.Scrollbar(gallery, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Collect one renderball per seed
        gallery._photos = []  # prevent GC
        col = 0
        row_idx = 0
        cols = 5
        thumb_size = 160

        seed_dirs = sorted([d for d in renders_dir.iterdir() if d.is_dir()])
        for seed_dir in seed_dirs:
            # Find the first renderball image
            rb_files = sorted(seed_dir.glob("*_renderball_*.png")) + \
                       sorted(seed_dir.glob("*_renderball_*.tga"))
            if not rb_files:
                continue
            try:
                img = Image.open(str(rb_files[0]))
                img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                gallery._photos.append(photo)

                cell = ttk.Frame(frame)
                cell.grid(row=row_idx, column=col, padx=4, pady=4)
                ttk.Label(cell, image=photo).pack()
                ttk.Label(cell, text=seed_dir.name, font=("", 8)).pack()

                col += 1
                if col >= cols:
                    col = 0
                    row_idx += 1
            except Exception:
                continue

        if not gallery._photos:
            ttk.Label(frame, text="No renderball images found.").pack(padx=20, pady=20)

    # ──────────────────────────────────────────────────────────────────────
    #  SBSAR Search
    # ──────────────────────────────────────────────────────────────────────
    def _search_sbsar(self):
        if self._running:
            return
        search_dir = filedialog.askdirectory(title="Select directory to search for .sbsar files")
        if not search_dir:
            return

        self._running = True
        self.search_status.config(text="Searching...")
        self.search_progress["value"] = 0
        self.root.update()

        def _do_search():
            try:
                sbsrender_exe = gen.find_sbsrender()
                if not sbsrender_exe:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "sbsrender not found \u2014 cannot validate .sbsar files"))
                    return

                # Find all .sbsar files
                all_sbsar = []
                for dirpath, _, filenames in os.walk(search_dir):
                    for fn in filenames:
                        if fn.lower().endswith(".sbsar"):
                            all_sbsar.append(os.path.join(dirpath, fn))

                total = len(all_sbsar)
                self.root.after(0, lambda: self.search_status.config(
                    text=f"Found {total} .sbsar files. Validating..."))

                # Load existing JSON to get known hashes
                existing = {}
                if os.path.isfile(SBSAR_JSON_PATH):
                    try:
                        with open(SBSAR_JSON_PATH, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for m in data.get("materials", []):
                            existing[m.get("hash", "")] = m
                    except Exception:
                        pass

                valid_materials = []
                seen_hashes = set(existing.keys())
                valid_count = 0

                for idx, sbsar_path in enumerate(all_sbsar):
                    if self._cancel:
                        break

                    # Compute hash for deduplication
                    file_hash = _file_hash(sbsar_path)
                    if file_hash in seen_hashes:
                        # Already in library, skip
                        pct = int((idx + 1) / total * 100)
                        self.root.after(0, lambda p=pct, v=valid_count: (
                            self.search_progress.configure(value=p),
                            self.search_status.config(
                                text=f"Checking {idx+1}/{total}  \u2022  {v} valid  \u2022  skipped duplicate")
                        ))
                        continue

                    # Validate: try a dry run render
                    is_valid = _validate_sbsar(sbsrender_exe, sbsar_path)
                    if is_valid:
                        valid_count += 1
                        seen_hashes.add(file_hash)
                        valid_materials.append({
                            "path": sbsar_path,
                            "name": Path(sbsar_path).stem,
                            "hash": file_hash,
                        })

                    pct = int((idx + 1) / total * 100)
                    status = "valid" if is_valid else "invalid"
                    self.root.after(0, lambda p=pct, v=valid_count, s=status, i=idx: (
                        self.search_progress.configure(value=p),
                        self.search_status.config(
                            text=f"Checking {i+1}/{total}  \u2022  {v} valid  \u2022  last: {s}")
                    ))

                # Merge with existing
                for m in valid_materials:
                    existing[m["hash"]] = m

                result = {"materials": list(existing.values())}
                with open(SBSAR_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                final_count = len(result["materials"])
                self.root.after(0, lambda: (
                    self.search_progress.configure(value=100),
                    self.search_status.config(
                        text=f"Done. {valid_count} new valid materials found. "
                             f"Library total: {final_count}"),
                    self._update_lib_label(),
                    self._update_total_label(),
                ))
            except Exception as e:
                self.root.after(0, lambda: self.search_status.config(text=f"Error: {e}"))
            finally:
                self._running = False
                self._cancel = False

        threading.Thread(target=_do_search, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    #  Generate Dataset
    # ──────────────────────────────────────────────────────────────────────
    def _generate_dataset(self):
        if self._running:
            return
        self._apply_settings()
        self._running = True
        self._cancel = False
        self.btn_generate.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.gen_progress["value"] = 0
        self.gen_status.config(text="Starting dataset generation...")
        self.root.update()

        def _do_generate():
            try:
                sbsrender_exe = gen.find_sbsrender()
                if not sbsrender_exe:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "sbsrender not found"))
                    return

                # Build list of (sbsar_path, num_seeds) jobs
                jobs = []
                seeds_per = self.v_seeds_per_mat.get()

                if self.v_use_main_sbsar.get():
                    for _ in range(gen.NUM_MATERIALS):
                        jobs.append(gen.SBSAR_PATH)

                if self.v_use_json_sbsar.get() and os.path.isfile(SBSAR_JSON_PATH):
                    with open(SBSAR_JSON_PATH, "r", encoding="utf-8") as f:
                        lib = json.load(f)
                    for mat in lib.get("materials", []):
                        jobs.append(mat["path"])

                total_jobs = len(jobs) * seeds_per
                if total_jobs == 0:
                    self.root.after(0, lambda: self.gen_status.config(text="Nothing to generate."))
                    return

                # Setup output
                renders_dir = Path(gen.OUTPUT_ROOT) / "renders"
                renders_dir.mkdir(parents=True, exist_ok=True)
                csv_path = Path(gen.OUTPUT_ROOT) / "dataset.csv"
                existing_rows, existing_seeds = gen.load_existing_csv(csv_path)
                all_rows = list(existing_rows)

                completed = 0
                succeeded = 0
                failed = 0

                for sbsar_path in jobs:
                    if self._cancel:
                        break

                    for _seed_i in range(seeds_per):
                        if self._cancel:
                            break

                        # Generate unique seed
                        seed = gen.generate_seed()
                        while seed in existing_seeds:
                            seed = gen.generate_seed()

                        # Determine output directory
                        is_library = (sbsar_path != gen.SBSAR_PATH)
                        if is_library:
                            mat_name = Path(sbsar_path).stem
                            out_dir = renders_dir / f"{mat_name}_{seed}"
                        else:
                            out_dir = renders_dir / seed

                        out_dir.mkdir(parents=True, exist_ok=True)

                        # Temporarily set SBSAR_PATH for this render
                        orig_sbsar = gen.SBSAR_PATH
                        gen.SBSAR_PATH = sbsar_path

                        # Also temporarily set OUTPUT_ROOT's seed_folder behavior
                        if is_library:
                            orig_seed_folder = gen.seed_folder
                            gen.seed_folder = lambda s, d=out_dir: d

                        ok = gen.render_material(sbsrender_exe, seed)
                        if ok:
                            channels = gen.discover_channels(seed)
                            found_ch = [k for k, v in channels.items() if v]

                            # Render shaderball previews
                            rb_paths = {}
                            for angle_cfg in gen.RENDERBALL_ANGLES:
                                label = angle_cfg["label"]
                                try:
                                    p = gen.render_sphere_preview(
                                        channels, seed,
                                        azimuth_deg=angle_cfg["azimuth_deg"],
                                        suffix=label)
                                    rb_paths[label] = p
                                except Exception:
                                    rb_paths[label] = ""

                            csv_dir = csv_path.parent
                            row = {
                                "seed": seed,
                                "folder_path": str(out_dir.relative_to(csv_dir)),
                            }
                            for k, v in channels.items():
                                try:
                                    row[k] = str(Path(v).relative_to(csv_dir)) if v else ""
                                except ValueError:
                                    row[k] = v
                            for lbl, p in rb_paths.items():
                                try:
                                    row[f"renderball_{lbl}"] = str(Path(p).relative_to(csv_dir)) if p else ""
                                except ValueError:
                                    row[f"renderball_{lbl}"] = p

                            all_rows.append(row)
                            existing_seeds.add(seed)
                            gen.write_csv(csv_path, all_rows)
                            succeeded += 1
                        else:
                            failed += 1
                            if out_dir.is_dir():
                                shutil.rmtree(out_dir, ignore_errors=True)

                        # Restore
                        gen.SBSAR_PATH = orig_sbsar
                        if is_library:
                            gen.seed_folder = orig_seed_folder

                        completed += 1
                        pct = int(completed / total_jobs * 100)
                        self.root.after(0, lambda p=pct, c=completed, t=total_jobs, s=succeeded, f=failed: (
                            self.gen_progress.configure(value=p),
                            self.gen_status.config(
                                text=f"{c}/{t}  \u2022  {s} succeeded  \u2022  {f} failed")
                        ))

                gen.write_csv(csv_path, all_rows)
                self.root.after(0, lambda: (
                    self.gen_status.config(
                        text=f"Done! {succeeded} succeeded, {failed} failed."),
                    self._refresh_csv_info(),
                ))
            except Exception as e:
                self.root.after(0, lambda: self.gen_status.config(text=f"Error: {e}"))
            finally:
                self._running = False
                self._cancel = False
                self.root.after(0, lambda: (
                    self.btn_generate.config(state="normal"),
                    self.btn_cancel.config(state="disabled"),
                ))

        threading.Thread(target=_do_generate, daemon=True).start()

    def _cancel_generation(self):
        self._cancel = True
        self.gen_status.config(text="Cancelling...")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           HELPER FUNCTIONS                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _file_hash(path: str) -> str:
    """SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sbsar(sbsrender_exe: str, sbsar_path: str) -> bool:
    """Check if sbsrender can render this .sbsar (quick info query)."""
    try:
        result = subprocess.run(
            [sbsrender_exe, "info", "--inputs", sbsar_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                               MAIN                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    root = tk.Tk()
    app = PBRDatasetGUI(root)
    root.mainloop()
