"""Evaluate a trained model on the held-out TEST split only.

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.

    python evaluate.py --weights runs/resnet18/best.pt
    python evaluate.py --weights release/model.safetensors
    python evaluate.py --weights runs/quick_test/best.pt --limit 1024     (code check only)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (DISCLAIMER, MEAN, STD, PCamDataset, eval_transform, get_device, labels_of,
                    load_model, load_split, safe_auc, seed_everything, tta_logits)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="best.pt from training, or model.safetensors")
    p.add_argument("--out", default="outputs")
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold (fixed in advance, default 0.5)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=None, help="evaluate a random subset (code check only, marks results as partial)")
    p.add_argument("--no-tta", action="store_true", help="turn off test-time augmentation (8 flips/rotations)")
    p.add_argument("--bootstrap", type=int, default=500, help="resamples for 95%% confidence intervals")
    return p.parse_args()


@torch.no_grad()
def run_model(model, rows, args, device):
    loader = DataLoader(PCamDataset(rows, eval_transform()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    mean = torch.tensor(MEAN).view(1, 3, 1, 1); std = torch.tensor(STD).view(1, 3, 1, 1)
    ps, ps_plain, bright = [], [], []
    for x, _ in tqdm(loader, desc="testing"):
        bright.append((x * std + mean).mean(dim=(1, 2, 3)))           # average pixel brightness 0..1
        xd = x.to(device)
        plain = torch.sigmoid(model(xd).squeeze(1)).cpu()
        ps_plain.append(plain)
        ps.append(torch.sigmoid(tta_logits(model, xd).squeeze(1)).cpu() if args.tta else plain)
    return torch.cat(ps).numpy(), torch.cat(ps_plain).numpy(), torch.cat(bright).numpy()


def basic_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = lambda a, b: float(a / b) if b else float("nan")
    prec, sens = d(tp, tp + fp), d(tp, tp + fn)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), accuracy=d(tp + tn, len(y)),
                sensitivity=sens, specificity=d(tn, tn + fp), precision=prec,
                f1=d(2 * prec * sens, prec + sens) if prec == prec and sens == sens else float("nan"),
                roc_auc=safe_auc(y, p))


def bootstrap_ci(y, p, thr, n, seed=0):
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m = basic_metrics(y[i], p[i], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, frac = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(frac - conf)
            rows.append(dict(bin=b, mean_confidence=conf, fraction_tumor=frac, count=int(m.sum())))
    return float(ece), rows


def fmt(v, ci=None):
    if v != v:
        return "n/a"
    return f"{v:.4f}" + (f" ({ci[0]:.4f} to {ci[1]:.4f})" if ci else "")


def main():
    args = parse_args()
    args.tta = not args.no_tta
    seed_everything(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, cfg = load_model(args.weights, device)

    print("Loading the official TEST split (only split this script can touch) ...")
    rows = load_split("test", args.limit, seed=0)
    y = labels_of(rows)
    p, p_plain, bright = run_model(model, rows, args, device)
    thr = args.threshold
    m = basic_metrics(y, p, thr)
    m_plain = basic_metrics(y, p_plain, thr) if args.tta else None
    ci = bootstrap_ci(y, p, thr, args.bootstrap)
    ece, cal_rows = calibration(y, p)
    brier = float(brier_score_loss(y, p))
    val_thr = cfg.get("threshold_val_tuned")
    m_val = basic_metrics(y, p, val_thr) if val_thr is not None else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"model (AUC {m['roc_auc']:.3f})"); ax.plot([0, 1], [0, 1], "--", c="gray", label="coin flip")
    ax.plot(1 - m["specificity"], m["sensitivity"], "ro", label=f"threshold {thr:g}")
    ax.set_xlabel("false positive rate (1 - specificity)"); ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title("ROC curve (test split)"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=130); plt.close(fig)

    # Confusion matrix
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred normal", "pred tumor"]); ax.set_yticks([0, 1], ["true normal", "true tumor"])
    ax.set_title(f"Confusion matrix (threshold {thr:g})"); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)

    # Reliability diagram
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot([0, 1], [0, 1], "--", c="gray", label="perfectly calibrated")
    ax[0].plot([r["mean_confidence"] for r in cal_rows], [r["fraction_tumor"] for r in cal_rows], "o-", label="model")
    ax[0].set_xlabel("predicted tumor probability"); ax[0].set_ylabel("actual fraction that is tumor"); ax[0].set_title(f"Calibration (ECE {ece:.3f})"); ax[0].legend()
    ax[1].hist(p, bins=20, range=(0, 1)); ax[1].set_xlabel("predicted tumor probability"); ax[1].set_ylabel("patches"); ax[1].set_title("How confident is the model?")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)

    # Failure cases: the most confidently wrong predictions
    pred = (p >= thr).astype(int)
    fp_idx = np.where((pred == 1) & (y == 0))[0]; fp_idx = fp_idx[np.argsort(-p[fp_idx])][:8]
    fn_idx = np.where((pred == 0) & (y == 1))[0]; fn_idx = fn_idx[np.argsort(p[fn_idx])][:8]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for r, (idxs, title) in enumerate([(fp_idx, "False positive (really normal)"), (fn_idx, "False negative (really tumor)")]):
        for c in range(8):
            a = axes[r, c]; a.axis("off")
            if c < len(idxs):
                a.imshow(rows[int(idxs[c])]["image"].convert("RGB")); a.set_title(f"#{idxs[c]}  p={p[idxs[c]]:.2f}", fontsize=9)
        axes[r, 0].text(-0.1, 0.5, title, rotation=90, va="center", ha="right", transform=axes[r, 0].transAxes, fontsize=9)
    fig.suptitle("Most confidently wrong test patches (p = predicted tumor probability)"); fig.tight_layout(); fig.savefig(out / "failure_cases.png", dpi=110); plt.close(fig)

    groups = {}
    for name, mask in [("true negative", (pred == 0) & (y == 0)), ("false positive", (pred == 1) & (y == 0)),
                       ("false negative", (pred == 0) & (y == 1)), ("true positive", (pred == 1) & (y == 1))]:
        groups[name] = dict(count=int(mask.sum()), mean_prob=float(p[mask].mean()) if mask.any() else None,
                            mean_brightness=float(bright[mask].mean()) if mask.any() else None)

    # A few RANDOM test examples for the demo (chosen without looking at the model's answers)
    ex = out / "examples"; ex.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for lab in (0, 1):
        for i in rng.choice(np.where(y == lab)[0], size=min(3, int((y == lab).sum())), replace=False):
            rows[int(i)]["image"].convert("RGB").save(ex / f"test_{int(i)}_true_{'tumor' if lab else 'normal'}.png")

    metrics = dict(weights=str(args.weights), arch=cfg["arch"], split="test", n=int(len(y)), n_tumor=int(y.sum()),
                   n_normal=int((1 - y).sum()), threshold=thr, partial=args.limit is not None, metrics=m,
                   ci95=ci, ece=ece, brier=brier, calibration_bins=cal_rows, outcome_groups=groups,
                   val_tuned_threshold=val_thr, metrics_at_val_tuned_threshold=m_val, tta=args.tta, metrics_without_tta=m_plain)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savetxt(out / "test_predictions.csv", np.c_[np.arange(len(y)), y, p], delimiter=",", header="index,label,prob_tumor", fmt=["%d", "%d", "%.5f"], comments="")

    L = []
    if metrics["partial"]:
        L.append(f"> WARNING: PARTIAL run on a random subset of {len(y):,} test patches. NOT the official result.\n")
    L += [f"Measured on the held-out **test split**: {len(y):,} patches ({metrics['n_tumor']:,} tumor, {metrics['n_normal']:,} normal). "
          f"Decision threshold {thr:g} (fixed in advance). 95% confidence intervals come from {args.bootstrap} bootstrap resamples.\n",
          "| Metric | Value (95% CI) |", "|---|---|"]
    for k, label in [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity (recall)"), ("specificity", "Specificity"),
                     ("precision", "Precision"), ("f1", "F1 score"), ("roc_auc", "ROC-AUC")]:
        L.append(f"| {label} | {fmt(m[k], ci[k])} |")
    L += [f"| Calibration error (ECE, lower is better) | {ece:.4f} |", f"| Brier score (lower is better) | {brier:.4f} |", "",
          f"Confusion matrix: true negatives {m['tn']:,}, false positives {m['fp']:,}, false negatives {m['fn']:,}, true positives {m['tp']:,}.\n"]
    if m_plain:
        L.append(f"Test-time augmentation (average over 8 flips/rotations of each patch) is ON for the numbers above. Without it: accuracy {m_plain['accuracy']:.4f}, "
                 f"sensitivity {m_plain['sensitivity']:.4f}, specificity {m_plain['specificity']:.4f}, ROC-AUC {m_plain['roc_auc']:.4f}.\n")
    if m_val:
        L.append(f"For information only: using the threshold {val_thr:.3f} picked on the validation split gives accuracy {m_val['accuracy']:.4f}, "
                 f"sensitivity {m_val['sensitivity']:.4f}, specificity {m_val['specificity']:.4f}.\n")
    table = "\n".join(L)
    (out / "results_table.md").write_text(table, encoding="utf-8")

    plain = """
## What each number means (plain English)
- **Accuracy**: out of all patches, the share the model got right. Can look good while hiding one kind of mistake.
- **Sensitivity (recall)**: out of patches that really contain tumor, how many the model caught. Low = tumors are being missed.
- **Specificity**: out of normal patches, how many the model correctly called normal. Low = many false alarms.
- **Precision**: when the model says "tumor", how often it is right.
- **F1**: one number that balances precision and sensitivity (high only if both are high).
- **ROC-AUC**: how well the model ranks tumor patches above normal ones, ignoring the threshold. 0.5 = coin flip, 1.0 = perfect.
- **Confusion matrix**: the four counts: right-normal, false alarms, missed tumors, right-tumor.
- **Calibration (ECE, Brier)**: whether "90% sure" really means right about 90% of the time. Lower error = more trustworthy confidence.
"""
    fail = "\n## Observed failure cases (measured, test split)\n| Outcome | Count | Mean predicted tumor prob | Mean brightness (0-1) |\n|---|---|---|---|\n"
    for k, g in groups.items():
        fail += f"| {k} | {g['count']:,} | {fmt(g['mean_prob']) if g['mean_prob'] is not None else 'n/a'} | {fmt(g['mean_brightness']) if g['mean_brightness'] is not None else 'n/a'} |\n"
    fail += ("\nThe image `failure_cases.png` shows the most confidently wrong patches. Only a pathologist can say WHY a patch fooled the model; "
             "these tables only report what was measured.\n")
    (out / "results.md").write_text(f"# Test results\n\n> {DISCLAIMER}\n\n{table}\n{plain}{fail}", encoding="utf-8")

    print("\n" + table + plain + fail)
    print(f"\nSaved plots, metrics.json and results.md to {out}/")
    if metrics["partial"]:
        print("PARTIAL RUN - do not publish these numbers.")


if __name__ == "__main__":
    main()
