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
