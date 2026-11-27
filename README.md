# PCam tumor-patch classifier (research and education only)

> **FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.**
> This model does not detect cancer in real patients. It was trained on one dataset (PatchCamelyon) and may
> perform worse on images from other hospitals or scanners.

A small PyTorch CNN (ResNet18 or EfficientNet-B0) that classifies 96x96 lymph node histopathology patches as
tumor / normal. Data: [PatchCamelyon](https://huggingface.co/datasets/1aurent/PatchCamelyon) (CC0 1.0), official
train / validation / test splits, never reshuffled.

## Files
| File | What it does |
|---|---|
| `train.py` | trains the model (AMP, light augmentation, early stopping, checkpoints, `--quick-test`, `--resume`) |
| `evaluate.py` | scores the **test split only**: accuracy, sensitivity, specificity, precision, F1, ROC-AUC, confusion matrix, ROC plot, calibration, failure cases |
| `predict.py` | classify your own 96x96 images from the command line |
| `export.py` | makes `.safetensors`, `.onnx`, the model card and the Space folder |
| `app.py` | Gradio demo with a visible disclaimer |
| `common.py` | shared data / model code |
| `model_card.md` | template for the Hugging Face model card (results are filled in by `export.py`) |
| `UPLOAD.md` | exact Hugging Face upload steps |

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt  # on a GPU PC install the CUDA build of torch first: https://pytorch.org/get-started/locally/
```

## 1. Quick test (any machine, minutes)
```
python train.py --quick-test
python evaluate.py --weights runs/quick_test/best.pt --limit 1024
```
Uses a tiny subset. The numbers mean nothing; it only proves the code runs.

## 2. Full training (needs a GPU: RTX 3050, Colab or Kaggle)
```
python train.py
python evaluate.py --weights runs/resnet18/best.pt
python export.py --checkpoint runs/resnet18/best.pt
```
- Out of GPU memory? `python train.py --batch-size 64`.
- Colab disconnected? `python train.py --resume`.
- Train from scratch (no ImageNet start): `python train.py --weights none`.
- First run downloads about 7.6 GB of data. `train.py` prints the time of the first epoch and an estimate for the whole run.

## Colab / Kaggle
Easiest: open `pcam_train_colab.ipynb` in Colab (File > Upload notebook) and run the cells in order. Manual version:
Turn on a GPU (Runtime > Change runtime type > T4 GPU), then in a cell:
```
!git clone https://github.com/NISTALTALSON/pcam-tumor-classifier
%cd pcam-tumor-classifier
!pip install -q -r requirements.txt
!python train.py
!python evaluate.py --weights runs/resnet18/best.pt
!python export.py --checkpoint runs/resnet18/best.pt
```
Download the `release/`, `space/` and `outputs/` folders afterwards (or zip them).

## Honesty rules
Only measured results are reported. Do not tune anything on the test split. See `model_card.md` for limitations.

## License
Code and weights MIT. Dataset CC0 1.0. Citations: Veeling et al. 2018 (PCam); Ehteshami Bejnordi et al. 2017 (Camelyon16).

## Getting the best results (what is in the recipe, and what is not)
Used, all with the official PCam splits only:
- Pretrained start (or `--weights none`), strong color/stain augmentation, flips and 90-degree turns
- Weight averaging (EMA), learning-rate warm-up plus cosine decay, up to 20 epochs with early stopping on validation AUC
- Test-time augmentation (average over 8 flips/rotations) in `evaluate.py`, `predict.py` and the demo. The test table also shows the numbers without it.
- Two architectures compared by **validation** AUC (`compare_runs.py`), and only the winner is evaluated on the test split

Deliberately not used:
- **More training data.** The official PCam train split is everything allowed under the official splits. Mixing in Camelyon17-WILDS (CC0, about 456k patches, five hospitals) could help with other hospitals, but its slides come from a different release and there is no slide-ID list to prove it does not overlap the PCam test slides. Leaking test slides would make the published numbers look better than they are. If you ever try it, do it as a separate, clearly labeled experiment.
- **Training on the validation split.** It is needed for honest early stopping and model choice.
- **Tuning anything on the test split.** The test split is looked at once, at the end.

Expect the validation score to be a bit higher than the test score. That gap is common on PCam and is not a bug.
