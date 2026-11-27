"""Gradio demo: upload a 96x96 histopathology patch, get a tumor probability.

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.
Run locally:  python app.py
"""
from pathlib import Path

import gradio as gr

from common import DISCLAIMER, IMG_SIZE, load_model
from predict import predict_prob

HERE = Path(__file__).parent
CANDIDATES = [HERE / "model.safetensors", HERE / "release" / "model.safetensors"]
WEIGHTS = next((c for c in CANDIDATES if c.exists()), None)
if WEIGHTS is None:
    raise SystemExit("model.safetensors not found. Run export.py first (or put it next to app.py).")
MODEL, CFG = load_model(WEIGHTS, "cpu")
THRESHOLD = float(CFG.get("threshold_default", 0.5))
TTA = bool(CFG.get("tta", True))

BANNER = f"## Research and education only - not for medical diagnosis\n**{DISCLAIMER}**"


def classify(img):
    if img is None:
        return None, "Please upload a patch image."
    note = ""
    if img.size != (IMG_SIZE, IMG_SIZE):
        note = (f"\n\nYour image is {img.size[0]}x{img.size[1]} and was resized to {IMG_SIZE}x{IMG_SIZE}. "
                "The model was trained on real 96x96 patches, so treat this result as unreliable.")
    p = predict_prob(MODEL, img, TTA)
    verdict = "Tumor tissue likely in the center region" if p >= THRESHOLD else "Normal tissue likely"
    text = (f"### {verdict}\nTumor probability: **{p:.1%}** (decision threshold {THRESHOLD:g}).{note}\n\n"
            "**Reminder: this is a research demo trained on one dataset (PatchCamelyon). "
            "It is not a medical device and must not be used for diagnosis.**")
    return {"Tumor tissue (positive)": p, "Normal tissue (negative)": 1 - p}, text


examples_dir = HERE / "examples"
examples = [[str(f)] for f in sorted(examples_dir.glob("*.png"))] if examples_dir.exists() else None

with gr.Blocks(title="PCam tumor-patch classifier (research only)") as demo:
    gr.Markdown(BANNER)
    gr.Markdown("Upload a **96x96** H&E lymph node patch. The label means: does the **center 32x32 pixel region** "
                "contain at least one tumor pixel? Example images are random PatchCamelyon test patches (CC0).")
    with gr.Row():
        inp = gr.Image(type="pil", label="96x96 patch")
        with gr.Column():
            lab = gr.Label(num_top_classes=2, label="Prediction")
            msg = gr.Markdown()
    gr.Button("Classify", variant="primary").click(classify, inp, [lab, msg])
    inp.change(classify, inp, [lab, msg])
    if examples:
        gr.Examples(examples=examples, inputs=inp)
    gr.Markdown(f"---\n{DISCLAIMER}")

if __name__ == "__main__":
    demo.launch()
