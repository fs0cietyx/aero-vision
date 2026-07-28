# Satellite Optical Cloud Removal & Semantic Synthesis

## Overview
This repository contains a production-grade machine learning pipeline for the removal of atmospheric obstruction (cloud cover) from high-resolution satellite imagery. Engineered specifically for compatibility with the ISRO LISS-IV sensor, the architecture leverages a deep convolutional neural network (ResNet34-UNet) to synthesize missing geographical data while maintaining rigorous spatial and radiometric accuracy.

## Architecture Highlights
- **Modified U-Net Structure:** Incorporates a 9-channel convolution layer adapted from standard ImageNet weights to accommodate complex geospatial tensors.
- **Cross-Sensor Inference:** Trained on 4-band Sentinel-2 data and synthetically mapped to 3-band LISS-IV inputs to generate true-color approximations.
- **Out-Of-Memory (OOM) Safe Execution:** Utilizes buffered sliding-window inference to process arbitrarily large GeoTIFF records without memory saturation.
- **Global Radiometric Normalization:** Applies sub-sampled 10-bit percentile stretching to ensure consistent terrain brightness across all tiles.

## Repository Structure
- `01_dataset_generation.py`: Retrieves training pairs via the Google Earth Engine API.
- `02_model_training.py`: Model definition, configuration, and training routine using L1 and Edge-loss constraints.
- `03_liss4_inference.py`: Production inference script for processing raw ISRO LISS-IV `.zip` archives into seamless cloud-free GeoTIFFs.
- `04_visual_validation.py`: Radiometrically matched visualization tools for manual validation.

## Usage Guidelines
1. Configure dependencies via `pip install -r requirements.txt`.
2. Generate baseline training data or supply compatible HDF5 archives.
3. Execute `02_model_training.py` specifying local hardware accelerators (CUDA/MPS recommended).
4. Utilize `03_liss4_inference.py` to process local LISS-IV archives. Ensure paths in the configuration block correspond to the target spatial files.
5. Validate outputs via `04_visual_validation.py`.

## License & Compliance
This software is provided as-is for research and hackathon demonstration purposes. Ensure compliance with data provider regulations (ISRO, Copernicus) when utilizing generated datasets.
