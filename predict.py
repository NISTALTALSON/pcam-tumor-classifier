"""Predict tumor vs. normal for one or more 96x96 patch images.

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.

    python predict.py patch1.png patch2.png --weights release/model.safetensors
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from common import DISCLAIMER, IMG_SIZE, eval_transform, load_model, tta_logits


def prepare(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        print(f"  note: image is {img.size}, resizing to {IMG_SIZE}x{IMG_SIZE}. Results are only reliable for real 96x96 patches.")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return eval_transform()(img)


@torch.no_grad()
def predict_prob(model, img: Image.Image, tta: bool = False) -> float:
    x = prepare(img).unsqueeze(0)
    logit = tta_logits(model, x) if tta else model(x)
    return float(torch.sigmoid(logit.squeeze()).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default="release/model.safetensors")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    model, cfg = load_model(a.weights, "cpu")
    tta = bool(cfg.get("tta", True))   # same setting that was used for the reported test results
    print(DISCLAIMER + "\n")
    for path in a.images:
        print(path)
        p = predict_prob(model, Image.open(path), tta)
        label = "tumor tissue likely in center region" if p >= a.threshold else "normal tissue likely"
        print(f"  tumor probability: {p:.3f}  ->  {label}\n")


if __name__ == "__main__":
    main()
