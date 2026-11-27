"""Compare finished full training runs by VALIDATION ROC-AUC (never by test) and pick the best one.

    python train.py                          # ResNet18 -> runs/resnet18
    python train.py --arch efficientnet_b0   # -> runs/efficientnet_b0
    python compare_runs.py                   # prints the table, writes runs/best.txt
"""
import json
from pathlib import Path

rows = []
for f in sorted(Path("runs").glob("*/train_summary.json")):
    s = json.loads(f.read_text(encoding="utf-8"))
    if not s.get("quick_test"):
        rows.append((s["best_val_auc"], f.parent, s))
if not rows:
    raise SystemExit("No finished full runs found in runs/ (quick tests are ignored).")

rows.sort(key=lambda r: -r[0])
print("run | arch | best validation AUC | best epoch | minutes")
for auc, d, s in rows:
    print(f"{d.name} | {s['args']['arch']} | {auc:.4f} | {s['best_epoch']} | {s['train_minutes']:.0f}")

best = (rows[0][1] / "best.pt").as_posix()
Path("runs/best.txt").write_text(best, encoding="utf-8")
print(f"\nBest by validation AUC: {best}\nNext:\n  python evaluate.py --weights {best}\n  python export.py --checkpoint {best}")
print("Only evaluate the winner on the test split once. Do not pick a model by looking at test numbers.")
