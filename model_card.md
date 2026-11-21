---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - medical
  - histopathology
  - digital-pathology
  - patchcamelyon
  - research-only
  - safetensors
  - onnx
datasets:
  - 1aurent/PatchCamelyon
---

# PCam tumor-patch classifier (research and education only)

> **FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.**
> This model does not detect cancer in real patients. It was trained on one public dataset and may perform
> worse on images from other hospitals, scanners or staining protocols.

## Model details
- **Task:** binary classification of 96x96 px H&E-stained lymph node histopathology patches.
- **Label meaning (PatchCamelyon):** positive = the **center 32x32 px region** contains at least one tumor pixel; tumor tissue only in the outer border does not count.
- **Architecture:** small ImageNet-style CNN (ResNet18 or EfficientNet-B0, see `config.json`) with a single-logit head. Apply a sigmoid to get a tumor probability.
- **Files:** `model.safetensors` (weights), `model.onnx` (if present), `config.json` (architecture, normalization, thresholds).
- **Code:** training, evaluation and demo code are in the project repository.

## Intended use
- Research, benchmarking, teaching and learning about deep learning for digital pathology.
- Reproducing a simple PatchCamelyon baseline.

## Out-of-scope use (do not do this)
- Diagnosis, screening, triage, treatment decisions or any clinical workflow.
- Analysis of real patient cases, or whole slides without independent validation.
- Other tissue types, cancer types, stains or magnifications than PatchCamelyon.

## Training data
- **PatchCamelyon (PCam)**, licensed **CC0 1.0** (public domain dedication), loaded from the Hugging Face mirror `1aurent/PatchCamelyon`.
- 327,680 color patches of 96x96 px extracted from the Camelyon16 whole-slide images of lymph node sections; the official split is 262,144 train / 32,768 validation / 32,768 test. The **official splits were used unchanged** (no slide overlap between splits).
- No non-commercial data and no patient-identifiable data were added by this project.
- **Pretraining note:** if `init_weights` in `config.json` is `imagenet`, training started from torchvision ImageNet weights. ImageNet's terms restrict the dataset itself to non-commercial research use; if that matters for you, check them or retrain with `--weights none`.

## Training procedure
<!--TRAINING-->

Augmentation: random flips, random 90-degree rotations, light color jitter. Decision threshold 0.5 unless stated.

## Results
<!--RESULTS-->

Plots in this repository: `roc_curve.png`, `confusion_matrix.png`, `calibration.png`, `failure_cases.png`. Every number above was measured on the held-out PCam **test split** by `evaluate.py`; none were estimated or copied from papers.

## Limitations and risks
- Trained and tested on **one dataset** (Camelyon16-derived, lymph node metastases of breast cancer, from two medical centers). Performance on other hospitals, scanners, stains or cancer types is unknown and may be much worse.
- Patch-level only: a label for a small image, not a slide-level or patient-level result.
- The label depends only on the center 32x32 region, so the model's task is narrower than "is there cancer in this image".
- Probabilities can be over- or under-confident; see the calibration results above.
- Label noise and annotation errors exist in any histopathology dataset; no pathologist reviewed this model's outputs.
- No fairness or subgroup analysis was done (patient attributes are not part of the dataset).
- PCam is a benchmark; strong benchmark scores do not imply clinical usefulness.

## Citations
- Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., Welling, M. (2018). *Rotation Equivariant CNNs for Digital Pathology.* MICCAI 2018. arXiv:1806.03962.
- Ehteshami Bejnordi, B., et al. (2017). *Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer.* JAMA 318(22):2199-2210. (Camelyon16)

```bibtex
@inproceedings{veeling2018rotation,
  title     = {Rotation Equivariant CNNs for Digital Pathology},
  author    = {Veeling, Bastiaan S. and Linmans, Jasper and Winkens, Jim and Cohen, Taco and Welling, Max},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2018}
}
@article{bejnordi2017camelyon16,
  title   = {Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer},
  author  = {Ehteshami Bejnordi, Babak and others},
  journal = {JAMA},
  volume  = {318},
  number  = {22},
  pages   = {2199--2210},
  year    = {2017}
}
```

## License
Code and weights: MIT. Dataset: CC0 1.0. This license is not a medical-device clearance.
