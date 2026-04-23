## Additional Sections

### HybridPBRDataset
CSV-driven dataset that collects per-sample PBR maps, renderballs, and conditioning scalars; returns (input_image, cond_tensor, pbr_targets).

### Refactored PBRNet
Now accepts image + conditioning scalars with cond_encoder, fuse layer, and fused features for texture and scalar heads.

### Training Pipeline
Includes LPIPS loss, updated PBR parameter registry with standardized keys, texture resolution increased to 256.

### Dataset Requirements
Dataset.csv must be generated using generate_pbr_dataset.py.