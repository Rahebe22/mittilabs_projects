# FBM_mittilabs
results from FBM

### Dataset Overview

This dataset consists of **38,332 georeferenced image chips**, each of size **224×224 pixels** with a spatial resolution of approximately **0.2986 meters per pixel**, captured in **EPSG:3857 (WGS 84 / Pseudo-Mercator)**. Each chip includes RGB bands and is paired with a corresponding label.

The image-label pairs originate from **7 large tiles**, each covering an area of **5×5 kilometers**. These were processed and re-tiled into smaller chips using the `retiled.ipynb` script included in the repository.

The labels are binary:
- **0**: Non-field
- **1**: Field  
These were collected as part of the **FBM project at MittiLabs company**.

The dataset is split into:
- **Training set**: 30,665 chips (80%)
- **Validation set**: 5,750 chips (15%)
- **Test set**: 1,917 chips (5%)

All images are stored in **Cloud-Optimized GeoTIFF (COG)** format and maintain accurate spatial metadata for downstream geospatial machine learning tasks.
