tb added 
catalog generator
google embedding downloader
model

data part
catalog.csv, and geojson files
bboxes of study area



# Rice-Field Segmentation (Sentinel-1) — Preprocessing & Training Notes

## Classes
Trained a **5-class semantic segmentation model**:

| ID | Class name              |
|----|-------------------------|
| 0  | non-field               |
| 1  | dry                     |
| 2  | flooded                 |
| 3  | saturated with water    |
| 4  | saturated without water |

- Pixels with weak/unknown labels use `-100` and are **ignored** during loss/metrics (**weakly supervised setting**).

### Label balance (raw counts + share of valid pixels)
**Total valid pixels (classes 0–4):** 17,934,745  

| ID | Class                  | Pixels     | Share of valid |
|----|------------------------|------------|----------------|
| 0  | non-field              | 16,930,792 | 94.41%         |
| 1  | dry                    | 70,867     | 0.40%          |
| 2  | flooded                | 262,062    | 1.46%          |
| 3  | saturated with water   | 169,422    | 0.94%          |
| 4  | saturated without water| 501,602    | 2.80%          |

- **Ignored pixels (-100):** 18,493,031 (~50.77% of all pixels before masking)  
- The dataset is **heavily imbalanced** toward class 0.

---

## Preprocessing

### NoData handling
- Any `NaN`/`Inf` in inputs is replaced with `-9999` and treated as NoData.

### Normalization
- **Per-band, per-tile min–max**  
  - `normal_strategy: "min_max"`, `stat_procedure: "lab"`  
- NoData (`[-9999]`) is excluded from statistics.

### Spatial consistency
- Train/validate/test chips: **224×224** pixels.  


---

## Loss: Locally-Weighted Tversky-Focal (with ignore masking)

To Directly addresses **extreme class imbalance** and focuses on **hard errors**.

- **Per-batch class weights:** Inverse frequency of present (non-ignored) labels, normalized.  
- Absent classes in a batch get a small floor weight (`1e-5`) to avoid NaNs.  
- **Ignore mask:** `ignore_index = -100`  
  - A binary valid mask is built from integer labels.  
  - Predictions and one-hot targets are multiplied by it so ignored pixels contribute **zero** to TP/FP/FN.

**Formula:**
\[
TI = \frac{TP + \text{smooth}}{TP + \alpha FN + \beta FP + \text{smooth}}
\]
\[
L = (1 - TI)^\gamma
\]
where:  
- \(\alpha = 0.7\)  
- \(\beta = 0.3\)  
- \(\gamma = 1.33\)

- Denominators are clamped and a small ε is added for stability.

---

## Optimizer & Learning Rate

- **Optimizer:** SAM (Sharpness-Aware Minimization) with **SGD** base optimizer, momentum = 0.95.  
To improve generalization under label noise and imbalance by preferring flatter minima.

- **LR schedule:** `CosineAnnealingWarmRestarts`  
  - `learning_rate_init: 0.01`  
  - `T_0: 10`, `T_mult: 2`, `eta_min: 1e-4`  
  - To smooth decay with periodic warm restarts retains progress after plateaus; converges faster than the 0.003 baseline.

- **Early stopping:**  
  - Patience = 100, Warm-up = 10 epochs.  
  - Ensures cosine schedule has time to work before stopping.

---

## Key Config (excerpt)
```yaml
Train_Validate:
  data_size: 224
  buffer: 0
  img_path_cols: ['image']
  label_path_col: 'label'

  apply_normalization: True
  normal_strategy: "min_max"
  stat_procedure: "lab"
  nodata: [-9999]

  n_classes: 5
  channels: 3
  ignore_index: -100
  criterion: LocallyWeightedTverskyFocalLoss()

  optimizer_name: SAM
  momentum: 0.95
  learning_rate_init: 0.01
  learning_rate_policy: "CosineAnnealingWarmRestarts"
  scheduler_kwargs:
    T_0: 10
    T_mult: 2
    eta_min: 0.0001

  train_batch: 32
  validate_batch: 10

  early_stopping_patience: 100
  warmup_period: 10
```
---

## Results

### Overall Metrics
| Metric           | Value   |
|------------------|---------|
| Overall Accuracy | 0.4955  |
| Mean Accuracy    | 0.3078  |
| Mean IoU         | 0.1170  |
| Mean Precision   | 0.2146  |
| Mean Recall      | 0.3078  |
| Mean F1 Score    | 0.1671  |

---

### Class-wise Metrics
| Class                  | Accuracy | IoU     | Precision | Recall  | F1 Score |
|------------------------|----------|---------|-----------|---------|----------|
| non-field              | 0.5031   | 0.4969  | 0.9756    | 0.5031  | 0.6639   |
| dry                    | 0.4380   | 0.0212  | 0.0218    | 0.4380  | 0.0416   |
| flooded                | 0.0114   | 0.0035  | 0.0050    | 0.0114  | 0.0070   |
| saturated with water   | 0.1098   | 0.0226  | 0.0276    | 0.1098  | 0.0441   |
| saturated without water| 0.4768   | 0.0410  | 0.0430    | 0.4768  | 0.0789   |

---

### Normalized Confusion Matrix
<img width="1789" height="1866" alt="finetune_metrics" src="https://github.com/user-attachments/assets/dafc8bf2-669d-4102-b10e-da8f65598279" />
---

### Training & Validation Loss
| Training Loss Curve | Validation Loss Curve |
|---------------------|-----------------------|

---

## Discussion

- **Overall Accuracy** reached ~49.55%, but the **Mean IoU** is low (0.117), reflecting challenges with minority classes.
- **Class 0 (non-field)** dominates performance due to class imbalance — achieving high precision (0.9756) and F1 score (0.6639).
- Minority classes (dry, flooded, saturated with/without water) show **low IoU and F1 scores**, suggesting further improvement is needed in recall and precision for these classes.
- The **confusion matrix** indicates:
  - Significant misclassification of flooded and saturated classes into *non-field* or *saturated without water*.
  - Dry is often confused with *saturated without water*.
- Loss curves suggest the model is learning but may be **plateauing early**, possibly due to label imbalance and noise.

---

## Next Steps
- Adding google embeddings as input
- Enhance **minority class representation** via targeted sampling or synthetic augmentation.
- Experiment with **multi-stage training** — first focusing on separating field vs. non-field, then sub-class segmentation.
- Explore **loss re-weighting** or **focal variants** with higher emphasis on flooded/saturated classes.
- Consider **temporal features** or multi-date stacks for better water-related class discrimination.

---
