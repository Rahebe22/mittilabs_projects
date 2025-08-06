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

### Model Overview

The semantic segmentation model used in this project is a **U-Net-based architecture**, developed by the **AIRG team at the Geography Department, Clark University**. The model is implemented in PyTorch and is tailored for binary segmentation of agricultural fields (non-field: 0, field: 1) using 224×224 multispectral image chips.

#### Architecture

The model follows a **deep U-Net design** with 6 encoding and decoding levels. Each block consists of configurable numbers of convolutional layers followed by BatchNorm and ReLU activations. Key architectural features include:

- **Filter configuration**: `[64, 128, 256, 512, 1024, 2048]`
- **Block depth**: `[2, 2, 2, 2, 2, 2]`
- **Dropout rate**: `0.1` using `"traditional"` dropout
- **Upsampling mode**: `'deconv_2'` (overlapping transposed convolution)
- **Skip attention**: Disabled (`use_skipAtt=False`)
- **Final classifier**: 1×1 convolution for 2 output classes

#### Training Configuration

- **Input channels**: 3 (RGB)
- **Number of classes**: 2
- **Loss function**: `LocallyWeightedTverskyFocalLoss()`
- **Optimizer**: `Sharpness-Aware Minimization (SAM)`
- **Initial Learning Rate**: `0.01` with `StepLR` scheduler
- **Epochs**: 80 (early stopping triggered epoch 29)
- **Batch size**: 32 (training), 10 (validation)
- **Early stopping**: patience of 10 epochs with `min_delta=0.001`
- **Warm-up period**: 10 epochs
- **Checkpointing**: every 20 epochs

#### Data Augmentation & Preprocessing

- **Normalization**: Min-Max scaling using label-specific stats
- **Augmentation**:
  - Scale jittering: `(0.75, 1.5)`
  - Random rotation: `(-180, -90, 90, 180)`
  - Gaussian blur: `σ ∈ [0.03, 0.07]`
  - Brightness shift: `[-0.02, 0.02]`
  - Contrast adjustment: `[0.9, 1.2]`
  - Gamma adjustment: `(0.2, 2.0)`
  - Patch shifting enabled

- **Wigth initialization**:
  - All model parameters are initialized using **Kaiming initialization**.


 ### Training results

![image](https://github.com/user-attachments/assets/80d95006-3994-47d4-b852-b8ccf5c6ef12)

**Overall Evaluation Metrics**

| **Metric**         | **Value**   |
|--------------------|-------------|
| Overall Accuracy   | 0.8114      |
| Mean Accuracy      | 0.8074      |
| Mean IoU           | 0.6792      |
| Mean Precision     | 0.8100      |
| Mean Recall        | 0.8074      |
| Mean F1 Score      | 0.8084      |


**Per-Class Evaluation Metrics**

| **Class**    | **Accuracy** | **IoU**  | **Precision** | **Recall** | **F1 Score** |
|--------------|--------------|----------|----------------|------------|--------------|
| non-field    | 0.7697       | 0.6454   | 0.7998         | 0.7697     | 0.7845       |
| field        | 0.8450       | 0.7129   | 0.8202         | 0.8450     | 0.8324       |


*Loss curve to be added*


 ### Prediction performance

 <img width="604" alt="Screenshot 2025-06-30 at 1 32 36 AM" src="https://github.com/user-attachments/assets/28a816e1-d21e-4583-8761-16e09ec71b0b" />

 **Prediction Overall Evaluation Metrics**

| **Metric**         | **Value**   |
|--------------------|-------------|
| Overall Accuracy   | 0.8107      |
| Mean Accuracy      | 0.8091      |
| Mean IoU           | 0.6793      |
| Mean Precision     | 0.8084      |
| Mean Recall        | 0.8091      |
| Mean F1 Score      | 0.8087      |


**Prediction Per-Class Evaluation Metrics**

| **Class**    | **Accuracy** | **IoU**    | **Precision** | **Recall** | **F1 Score** |
|--------------|--------------|------------|----------------|------------|--------------|
| non-field    | 0.7944       | 0.6517     | 0.7840         | 0.7944     | 0.7891       |
| field        | 0.8239       | 0.7070     | 0.8328         | 0.8239     | 0.8283       |


**Predicted tile examples**

![image](https://github.com/user-attachments/assets/55d042f1-c93a-4c3b-8505-c190003e3993)


