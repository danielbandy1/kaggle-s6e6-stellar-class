#!/usr/bin/env python3
"""Blend LGB + XGB probabilities for final ensemble submission.

Requires:
  models/lgb_oof_proba.npy  (from feateng_lgb.py — saved manually or added to that script)
  models/lgb_test_proba.npy
  models/xgb_oof_proba.npy  (from xgb_model.py)
  models/xgb_test_proba.npy
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/home/daniel/kaggle/s6e6")
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
COMP = "playground-series-s6e6"


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train["class"].astype(str))

    # Load probas
    lgb_oof = np.load(MODELS / "lgb_oof_proba.npy")
    lgb_test = np.load(MODELS / "lgb_test_proba.npy")
    xgb_oof = np.load(MODELS / "xgb_oof_proba.npy")
    xgb_test = np.load(MODELS / "xgb_test_proba.npy")

    results = []
    best_w, best_acc = 0.5, 0.0
    for w in np.arange(0.3, 0.8, 0.05):
        blend_oof = w * lgb_oof + (1 - w) * xgb_oof
        acc = accuracy_score(y, blend_oof.argmax(axis=1))
        f1m = f1_score(y, blend_oof.argmax(axis=1), average="macro")
        results.append({"lgb_weight": round(float(w), 2), "oof_accuracy": acc, "oof_f1_macro": f1m})
        print(f"lgb_w={w:.2f} accuracy={acc:.6f} f1_macro={f1m:.6f}")
        if acc > best_acc:
            best_acc, best_w = acc, w

    print(f"\nBest blend: lgb_weight={best_w:.2f} OOF accuracy={best_acc:.6f}")
    blend_test = best_w * lgb_test + (1 - best_w) * xgb_test

    pred_labels = target_encoder.inverse_transform(blend_test.argmax(axis=1))
    sub = sample.copy()
    sub.iloc[:, 1] = pred_labels
    sub_path = SUBMISSIONS / f"blend_lgb{int(best_w*100)}_xgb{int((1-best_w)*100)}_submission.csv"
    sub.to_csv(sub_path, index=False)

    (MODELS / "blend_metrics.json").write_text(
        json.dumps({"best_lgb_weight": best_w, "best_oof_accuracy": best_acc,
                    "grid_results": results}, indent=2))

    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(sub_path),
                    "-m", f"LGB+XGB blend w={best_w:.2f} feateng 10fold"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
