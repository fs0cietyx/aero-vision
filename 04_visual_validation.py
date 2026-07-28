"""
Phase 4: Visual Validation
Provides OOM-safe, globally normalized visual validation of the reconstructed imagery 
against the raw ISRO LISS-IV observations.
"""
import glob
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_TIF = "ISRO_LISS4_CloudFree_Seamless.tif"
EXTRACT_DIR = "LISS4_RAW"
ZOOM_FACTOR = 4 

def main():
    print(f"Loading primary observation record (Scaled down {ZOOM_FACTOR}x for memory safety)...")
    band2_path = glob.glob(f"{EXTRACT_DIR}/**/*BAND2*.tif", recursive=True)[0]
    band3_path = glob.glob(f"{EXTRACT_DIR}/**/*BAND3*.tif", recursive=True)[0]

    with rasterio.open(band3_path) as src:
        H, W = src.height, src.width
        out_shape = (int(H / ZOOM_FACTOR), int(W / ZOOM_FACTOR))
        raw_red = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)

    with rasterio.open(band2_path) as src:
        raw_green = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)

    raw_rgb = np.stack([raw_red, raw_green, raw_green], axis=-1)
    del raw_red, raw_green 

    print("Loading AI-reconstructed spatial record...")
    with rasterio.open(OUTPUT_TIF) as src:
        out_red = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
        out_green = src.read(2, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
        out_blue = src.read(3, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)

    out_rgb = np.stack([out_red, out_green, out_blue], axis=-1)
    del out_red, out_green, out_blue 

    print("Applying global radiometric alignment...")
    c_min, c_max = np.percentile(out_rgb[::5, ::5], (2, 98))

    raw_display = np.clip((raw_rgb - c_min) / (c_max - c_min + 1e-8), 0, 1)
    out_display = np.clip((out_rgb - c_min) / (c_max - c_min + 1e-8), 0, 1)

    print("Rendering validation graphics...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.patch.set_facecolor('#1e1e1e')

    axes[0].imshow(raw_display)
    axes[0].set_title("Primary Observation: ISRO LISS-IV (Obscured)", fontsize=22, fontweight='bold', color='white', pad=20)
    axes[0].axis("off")

    axes[1].imshow(out_display)
    axes[1].set_title("Reconstruction: Synthetic Baseline (Clear)", fontsize=22, fontweight='bold', color='#00ff88', pad=20)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("validation_result.png", dpi=150, facecolor='#1e1e1e')
    print("Validation artifact saved as validation_result.png")

if __name__ == "__main__":
    main()
