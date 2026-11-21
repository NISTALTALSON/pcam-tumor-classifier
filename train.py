"""Train a small CNN to tell tumor from normal tissue in 96x96 PCam patches.

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.

Quick check (tiny subset, ~minutes even on CPU):   python train.py --quick-test
Full run (GPU):                                    python train.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (PCamDataset, build_model, eval_transform, get_device, labels_of,
                    load_split, safe_auc, seed_everything, train_transform)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="'imagenet' = pretrained start, 'none' = train from scratch")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128, help="use 64 if you run out of GPU memory")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3, help="early stopping: epochs without val-AUC gain")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="output folder (default runs/<arch>)")
    p.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--allow-cpu", action="store_true", help="allow a full run on CPU (very slow)")
    p.add_argument("--quick-test", action="store_true", help="tiny subset, 2 epochs, just to check the code")
    return p.parse_args()


@torch.no_grad()
def predict_loader(model, loader, device, use_amp):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logit = model(x).squeeze(1)
        ps.append(torch.sigmoid(logit.float()).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def torch_save_safe(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.quick_test:
        args.epochs, args.workers, args.batch_size = 2, 0, 64
        n_train, n_val = 2048, 512
    else:
        n_train = n_val = None
    out = Path(args.out or ("runs/quick_test" if args.quick_test else f"runs/{args.arch}"))
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cpu" and not (args.quick_test or args.allow_cpu):
        raise SystemExit("No GPU found. A full run on CPU would take days. Use the RTX 3050 PC, "
                         "Colab or Kaggle (see README), or add --allow-cpu if you really want it.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  mixed precision: {use_amp}")
    else:
        print("Running on CPU (fine for --quick-test only).")

    print("Loading official train / validation splits ...")
    train_rows = load_split("train", n_train, args.seed)
    val_rows = load_split("val", n_val, args.seed)
    ytr = labels_of(train_rows)
    print(f"train: {len(train_rows):,} patches ({ytr.mean() * 100:.1f}% tumor) | validation: {len(val_rows):,}")

    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    if args.workers > 0:
        kw["prefetch_factor"] = 4
    train_loader = DataLoader(PCamDataset(train_rows, train_transform()), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PCamDataset(val_rows, eval_transform()), batch_size=256, shuffle=False, **kw)

    model = build_model(args.arch, args.weights).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.BCEWithLogitsLoss()

    start, best_auc, bad, history = 0, -1.0, 0, []
    last_path, best_path = out / "last.pt", out / "best.pt"
    if args.resume and last_path.exists():
        s = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, best_auc, bad, history = s["epoch"] + 1, s["best_auc"], s["bad"], s["history"]
        print(f"Resumed from epoch {start}")

    t_start = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(x).squeeze(1)
            loss = crit(logit.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
        sched.step()

        yv, pv = predict_loader(model, val_loader, device, use_amp)
        row = dict(epoch=epoch + 1, train_loss=run_loss / seen, val_loss=bce(yv, pv),
                   val_acc=float(((pv >= 0.5) == yv).mean()), val_auc=safe_auc(yv, pv),
                   seconds=time.time() - t0)
        history.append(row)
        print(f"epoch {row['epoch']:>2} | train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} "
              f"| val acc {row['val_acc']:.4f} | val AUC {row['val_auc']:.4f} | {row['seconds']:.0f}s")
        if epoch == start and device.type == "cuda":
            print(f"   peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"   estimated time for all {args.epochs} epochs: ~{row['seconds'] * args.epochs / 60:.0f} min "
                  "(early stopping may finish sooner)")

        improved = row["val_auc"] > best_auc + 1e-4
        if improved:
            best_auc, bad = row["val_auc"], 0
            torch_save_safe(dict(model=model.state_dict(), arch=args.arch, init_weights=args.weights,
                                 epoch=epoch + 1, val_auc=best_auc), best_path)
        else:
            bad += 1
        torch_save_safe(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                             scaler=scaler.state_dict(), epoch=epoch, best_auc=best_auc, bad=bad,
                             history=history), last_path)
        if bad >= args.patience:
            print(f"Early stopping: no validation-AUC gain for {args.patience} epochs.")
            break

    # Pick a decision threshold on the VALIDATION set (never on test) and store it for reference.
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    yv, pv = predict_loader(model, val_loader, device, use_amp)
    thr = 0.5
    if len(np.unique(yv)) == 2:
        fpr, tpr, th = roc_curve(yv, pv)
        j = np.argmax(tpr - fpr)
        thr = float(np.clip(th[j], 0.0, 1.0))
    best["val_threshold"] = thr
    torch_save_safe(best, best_path)

    summary = dict(args=vars(args), best_epoch=best["epoch"], best_val_auc=best["val_auc"],
                   val_threshold=thr, epochs_run=len(history), history=history,
                   train_minutes=(time.time() - t_start) / 60, n_train=len(train_rows), n_val=len(val_rows),
                   train_pos_rate=float(ytr.mean()), quick_test=args.quick_test,
                   device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train"); ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_auc"] for h in history]); ax[1].set_title("validation ROC-AUC"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out / "history.png", dpi=120)
    print(f"\nDone. Best epoch {best['epoch']} (val AUC {best['val_auc']:.4f}). Checkpoint: {best_path}")
    if args.quick_test:
        print("QUICK TEST ONLY - these numbers are meaningless; it just proves the code runs.")


if __name__ == "__main__":
    main()
