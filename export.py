"""Package a trained model for release: .safetensors, ONNX, config, model card, and Space folder.

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.

    python export.py --checkpoint runs/resnet18/best.pt
Needs outputs/metrics.json from evaluate.py (a full test-split run). Refuses to package partial results.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from common import DATASET_ID, DISCLAIMER, IMG_SIZE, MEAN, STD, load_model

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-dir", default="outputs")
    ap.add_argument("--release", default="release")
    ap.add_argument("--space", default="space")
    ap.add_argument("--force", action="store_true", help="package even if evaluation is missing/partial (never publish that)")
    a = ap.parse_args()

    ev = Path(a.eval_dir)
    mj = ev / "metrics.json"
    metrics = json.loads(mj.read_text(encoding="utf-8")) if mj.exists() else None
    if (metrics is None or metrics.get("partial")) and not a.force:
        raise SystemExit("No full test-split evaluation found. Run evaluate.py on the full test split first "
                         "(or use --force for a local test package that must NOT be published).")

    ckpt_path = Path(a.checkpoint)
    model, cfg = load_model(ckpt_path, "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    summary_path = ckpt_path.parent / "train_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    rel = Path(a.release); rel.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(rel / "model.safetensors"), metadata={"format": "pt"})
    config = dict(arch=cfg["arch"], img_size=IMG_SIZE, mean=list(MEAN), std=list(STD), num_logits=1,
                  labels={"0": "normal", "1": "tumor (center 32x32 contains tumor)"}, threshold_default=0.5, tta=bool(metrics.get("tta", False)) if metrics else False,
                  threshold_val_tuned=ck.get("val_threshold"), init_weights=ck.get("init_weights"),
                  dataset=DATASET_ID, disclaimer=DISCLAIMER)
    (rel / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ONNX (optional; skipped with a message if the exporter fails)
    try:
        import onnxruntime as ort
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        onnx_path = rel / "model.onnx"
        kw = dict(input_names=["image"], output_names=["logit"], dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}})
        try:
            torch.onnx.export(model, dummy, str(onnx_path), opset_version=17, dynamo=False, **kw)
        except Exception:
            torch.onnx.export(model, dummy, str(onnx_path), opset_version=18, **kw)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
        diff = float(np.abs(sess.run(None, {"image": x.numpy()})[0].reshape(-1) - model(x).detach().numpy().reshape(-1)).max())
        print(f"ONNX exported. Max logit difference vs PyTorch: {diff:.2e}")
        if diff > 1e-3:
            print("WARNING: ONNX output differs from PyTorch more than expected - do not publish the ONNX file.")
    except Exception as e:
        print(f"ONNX export skipped: {e}")
        (rel / "model.onnx").unlink(missing_ok=True)

    # Model card: fill the placeholders with MEASURED results only
    card = (HERE / "model_card.md").read_text(encoding="utf-8")
    results = (ev / "results_table.md").read_text(encoding="utf-8") if (ev / "results_table.md").exists() else "NOT EVALUATED YET."
    if summary:
        s = summary
        training = (f"- Architecture: {cfg['arch']} (single-logit head), start weights: {ck.get('init_weights')}\n"
                    f"- Epochs run: {s['epochs_run']} (best epoch {s['best_epoch']}, validation ROC-AUC {s['best_val_auc']:.4f}); early stopping on validation AUC\n"
                    f"- Optimizer AdamW, lr {s['args']['lr']}, weight decay {s['args']['weight_decay']}, batch size {s['args']['batch_size']}, mixed precision, seed {s['args']['seed']}\n"
                    f"- Augmentation: {s['args'].get('aug', 'light')} color/stain jitter + flips + 90-degree turns; weight averaging (EMA) decay {s['args'].get('ema_decay', 0)}; LR warm-up then cosine decay; test-time augmentation: {'on' if config['tta'] else 'off'}\n"
                    f"- Train/validation patches: {s['n_train']:,} / {s['n_val']:,}; hardware: {s['device']}; training time {s['train_minutes']:.0f} min")
    else:
        training = "Training summary not found."
    card = card.replace("<!--TRAINING-->", training).replace("<!--RESULTS-->", results)
    (rel / "README.md").write_text(card, encoding="utf-8")
    for f in ("roc_curve.png", "confusion_matrix.png", "calibration.png", "failure_cases.png", "results.md"):
        if (ev / f).exists():
            shutil.copy(ev / f, rel / f)
    print(f"Model repo folder ready: {rel}/")

    # Space folder (self-contained demo)
    sp = Path(a.space)
    if sp.exists():
        shutil.rmtree(sp)
    sp.mkdir(parents=True)
    for f in ("app.py", "common.py", "predict.py"):
        shutil.copy(HERE / f, sp / f)
    shutil.copy(rel / "model.safetensors", sp / "model.safetensors"); shutil.copy(rel / "config.json", sp / "config.json")
    if (ev / "examples").exists():
        shutil.copytree(ev / "examples", sp / "examples")
    import gradio
    (sp / "requirements.txt").write_text("torch\ntorchvision\nsafetensors\npillow\nnumpy\n", encoding="utf-8")
    (sp / "README.md").write_text(
        "---\ntitle: PCam Tumor Patch Classifier (Research Only)\nemoji: \U0001F52C\ncolorFrom: blue\ncolorTo: purple\n"
        f"sdk: gradio\nsdk_version: {gradio.__version__}\napp_file: app.py\npinned: false\nlicense: mit\n---\n\n"
        f"# PCam tumor-patch classifier (demo)\n\n**{DISCLAIMER}**\n", encoding="utf-8")
    print(f"Space folder ready: {sp}/")


if __name__ == "__main__":
    main()
