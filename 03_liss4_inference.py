"""
Phase 3: Cross-Sensor Inference (LISS-IV)
Applies the Sentinel-2 trained model to LISS-IV imagery using a 
memory-safe sliding window approach and pseudo-true-color channel mapping.
"""
import os
import glob
import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

# Configuration
ZIP_PATH = "R2F20JUL2023063576009400060SSANSTUC00GTDB.zip"
EXTRACT_DIR = "LISS4_RAW"
WEIGHTS_PATH = "isro_model_epoch_25.pth"
OUTPUT_TIF = "ISRO_LISS4_CloudFree_Seamless.tif"

STEP = 2048
PADDING = 256

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AdvancedCloudUNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(9, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False, padding_mode='reflect')

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.dec4 = DecoderBlock(512 + 256, 256)
        self.dec3 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_mp = self.maxpool(x0)

        s1 = self.layer1(x_mp)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, x0)

        return self.final_up(d1)

def main():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if not glob.glob(f"{EXTRACT_DIR}/**/*BAND2*.tif", recursive=True):
        print(f"Extracting raw spatial data from {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    band2_path = glob.glob(f"{EXTRACT_DIR}/**/*BAND2*.tif", recursive=True)[0]
    band3_path = glob.glob(f"{EXTRACT_DIR}/**/*BAND3*.tif", recursive=True)[0]

    print("Calculating global radiometric percentiles (OOM-Safe Downsampling)...")
    with rasterio.open(band2_path) as src2, rasterio.open(band3_path) as src3:
        H, W = src2.height, src2.width
        b2_small = src2.read(1, out_shape=(H//10, W//10), resampling=Resampling.nearest).astype(np.float32)
        b3_small = src3.read(1, out_shape=(H//10, W//10), resampling=Resampling.nearest).astype(np.float32)

    image_rgb_small = np.stack([b3_small, b2_small, b2_small], axis=0)
    c_min = np.percentile(image_rgb_small, 2)
    c_max = np.percentile(image_rgb_small, 98)
    print(f"Global bounds detected -> Min: {c_min:.1f}, Max: {c_max:.1f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCloudUNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    with rasterio.open(band2_path) as ref:
        meta = ref.meta.copy()

    meta.update(count=3, dtype=rasterio.uint16)

    print("Initiating cross-sensor true-color synthesis via sliding window...")
    with rasterio.open(band2_path) as src2, \
         rasterio.open(band3_path) as src3, \
         rasterio.open(OUTPUT_TIF, 'w', **meta) as dst:

        with torch.no_grad():
            for y in range(0, H, STEP):
                for x in range(0, W, STEP):
                    write_w = min(STEP, W - x)
                    write_h = min(STEP, H - y)
                    write_window = Window(x, y, write_w, write_h)

                    read_x = x - PADDING
                    read_y = y - PADDING
                    read_window = Window(read_x, read_y, STEP + 2 * PADDING, STEP + 2 * PADDING)

                    b2_patch = src2.read(1, window=read_window, boundless=True, fill_value=0).astype(np.float32)
                    b3_patch = src3.read(1, window=read_window, boundless=True, fill_value=0).astype(np.float32)

                    patch = np.stack([b2_patch, b2_patch, b3_patch, b2_patch], axis=0)
                    patch_norm = np.clip((patch - c_min) / (c_max - c_min + 1e-8), 0.0, 1.0)

                    patch_t = torch.tensor(patch_norm).unsqueeze(0).to(device)
                    mask_t = torch.zeros((1, 1, b2_patch.shape[0], b2_patch.shape[1])).to(device)
                    temp_t = patch_t.clone()

                    inputs = torch.cat([patch_t, mask_t, temp_t], dim=1)
                    pred = model(inputs).cpu().squeeze(0).numpy()

                    pred_bands = pred[[2, 1, 0], :, :]
                    pred_scaled = np.clip(pred_bands * (c_max - c_min) + c_min, 0, 65535).astype(np.uint16)

                    out_crop = pred_scaled[:, PADDING : PADDING + write_h, PADDING : PADDING + write_w]
                    dst.write(out_crop, window=write_window)

                print(f"Processed geospatial coordinate row {y}/{H}")

    print(f"Analysis-ready GeoTIFF generated: {OUTPUT_TIF}")

if __name__ == "__main__":
    main()
