# Model Card: Lung X-ray Classifier

## Model details
- **Task**: 3-class image classification of frontal chest radiographs: `COVID19`, `NORMAL`, `PNEUMONIA`.
- **Architecture**: ImageNet-pretrained ResNet-18 (deployed; chosen for highest disease sensitivity). EfficientNet-B0, DenseNet-121,
  ConvNeXt-Tiny and the original 11.7K-parameter custom CNN are also supported. Fine-tuned end to end.
- **Outputs**: temperature-calibrated class probabilities, a `needs_review` flag (confidence below
  the referral threshold, default 0.70), and an optional Grad-CAM heatmap.
- **Artifact**: a single self-describing bundle (`model.pt`) holding weights, preprocessing,
  classes, temperature, evaluation summary and drift reference statistics.

## Intended use
- Education, research and portfolio demonstration of a responsible medical-imaging ML workflow.
- **Out of scope**: clinical diagnosis, triage or any decision about a real patient. The model
  has not been validated on an independent population and is not a regulated medical device.

## Training data
[Chest X-ray COVID-19 & Pneumonia (Kaggle, prashant268)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia):
5,144 train and 1,288 test images, aggregated from several public sources.

### Audit results (`lungxray audit`)
| Check | Result |
|---|---|
| Images in near-duplicate groups (pHash distance ≤ 4) | 323 |
| Test images with a near-duplicate in train | 45 (3.5%), excluded from the clean test set |
| Duplicate groups with conflicting labels | 19, removed |
| Balanced accuracy of a classifier on **file metadata only** | 0.874 (chance 0.333) |
| Median resolution: Normal / Pneumonia / COVID-19 | 1615×1322 / 1144×776 / 1165×1028 |

**Interpretation**: the classes come from different acquisition sources, so non-clinical cues
(resolution, aspect ratio, compression, text markers) correlate strongly with the label.
Mitigations: grayscale conversion, center-crop preprocessing (removes aspect ratio), duplicate-aware
splitting, and Grad-CAM review of failure cases. These **reduce but cannot eliminate** shortcut
learning, so the reported test performance is likely **optimistic** relative to a new hospital.

## Evaluation
- Clean held-out test set, with near-duplicates of training images removed.
- Metrics: balanced accuracy (headline), per-class sensitivity, specificity and AUC, macro F1,
  "abnormal vs normal" screening sensitivity, expected calibration error, and 95% bootstrap CIs.
- Full per-run results: `artifacts/runs/<run>/metrics.json`. Summary table in the README.

## Ethical considerations and limitations
- **Missed disease is the costly error.** Monitor `abnormal_sensitivity` and `missed_disease_cases`,
  not accuracy.
- No patient IDs, so leakage between images of the same patient can only be approximated.
- No demographic metadata, so performance across age, sex or device cannot be assessed.
- COVID-19 images in public datasets are often from severe, hospitalised cases (spectrum bias).
- Grad-CAM is a qualitative tool. A plausible heatmap does not prove correct reasoning.

## Recommended next steps
1. External validation on an independent dataset (RSNA Pneumonia, NIH ChestX-ray14 subset).
2. Lung-field segmentation to mask non-anatomical regions before classification.
3. Subgroup analysis once demographic metadata is available.
