#!/usr/bin/env python3
"""DART LGB for S6E6 — Dropout Additive Regression Trees.

DART (Rashmi & Gilad-Bachrach 2015) applies dropout to boosted trees,
reducing correlation between trees and often improving generalization.
Key params: drop_rate=0.1-0.2, skip_drop=0.5, max_drop=50.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/home/daniel/kaggle/s6e6")
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
COMP = "playground-series-s6e6"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u_minus_g"] = df["u"] - df["g"]
    df["g_minus_r"] = df["g"] - df["r"]
    df["r_minus_i"] = df["r"] - df["i"]
    df["i_minus_z"] = df["i"] - df["z"]
    df["u_minus_r"] = df["u"] - df["r"]
    df["g_minus_z"] = df["g"] - df["z"]
    df["u_minus_z"] = df["u"] - df["z"]
    df["g_minus_i"] = df["g"] - df["i"]
    df["r_minus_z"] = df["r"] - df["z"]
    df["u_minus_i"] = df["u"] - df["i"]
    df["color_curve_ugr"] = (df["u"] - df["g"]) - (df["g"] - df["r"])
    df["color_curve_gri"] = (df["g"] - df["r"]) - (df["r"] - df["i"])
    df["spectral_slope"] = (df["u"] - df["z"]) / 4.0
    df["log_redshift"] = np.log1p(df["redshift"].clip(lower=0))
    df["redshift_sq"] = df["redshift"] ** 2
    df["redshift_cube"] = df["redshift"] ** 3
    df["is_high_z"] = (df["redshift"] > 0.5).astype(np.float32)
    df["is_zero_z"] = (df["redshift"].abs() < 0.01).astype(np.float32)
    df["z_x_gmr"] = df["redshift"] * (df["g"] - df["r"])
    df["z_x_umg"] = df["redshift"] * (df["u"] - df["g"])
    df["z_x_rmi"] = df["redshift"] * (df["r"] - df["i"])
    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"] = mags.mean(axis=1)
    df["mag_std"] = mags.std(axis=1)
    df["mag_range"] = mags.max(axis=1) - mags.min(axis=1)
    df["cos_alpha"] = np.cos(np.radians(df["alpha"]))
    df["sin_alpha"] = np.sin(np.radians(df["alpha"]))
    df["cos_delta"] = np.cos(np.radians(df["delta"]))
    df["sin_delta"] = np.sin(np.radians(df["delta"]))
    return df


def preprocess(train_x, test_x):
    train_x = add_features(train_x)
    test_x = add_features(test_x)
    cat_cols = [c for c in train_x.columns if train_x[c].dtype == "object"]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_x[col], test_x[col]]).astype(str).fillna("__MISSING__")
        le.fit(combined)
        train_x[col] = le.transform(train_x[col].astype(str).fillna("__MISSING__"))
        test_x[col] = le.transform(test_x[col].astype(str).fillna("__MISSING__"))
    for col in train_x.columns:
        if train_x[col].isna().any() or test_x[col].isna().any():
            m = train_x[col].median()
            train_x[col] = train_x[col].fillna(0 if pd.isna(m) else m)
            test_x[col] = test_x[col].fillna(0 if pd.isna(m) else m)
    return train_x, test_x


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    id_col, target = "id", "class"
    feature_cols = [c for c in train.columns if c not in (target, id_col)]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train[target].astype(str))
    classes = list(target_encoder.classes_)
    n_classes = len(classes)

    x_train, x_test = preprocess(train[feature_cols], test[feature_cols])
    print(f"Features: {x_train.shape[1]}  Classes: {classes}")

    # DART-specific params: no early stopping (DART doesn't support it well)
    params = {
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "boosting_type": "dart",
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "max_drop": 50,
        "uniform_drop": False,
        "objective": "multiclass",
        "num_class": n_classes,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    oof_proba = np.zeros((len(train), n_classes))
    test_proba = np.zeros((len(test), n_classes))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), 1):
        model = LGBMClassifier(**params)
        model.fit(x_train.iloc[tr_idx], y[tr_idx])  # no early stopping for DART
        va_p = model.predict_proba(x_train.iloc[va_idx])
        te_p = model.predict_proba(x_test)
        oof_proba[va_idx] = va_p
        test_proba += te_p / skf.get_n_splits()
        acc = accuracy_score(y[va_idx], va_p.argmax(axis=1))
        f1m = f1_score(y[va_idx], va_p.argmax(axis=1), average="macro")
        fold_rows.append({"fold": fold, "accuracy": acc, "f1_macro": f1m})
        print(f"fold={fold:2d} accuracy={acc:.6f} f1_macro={f1m:.6f}")

    oof_acc = accuracy_score(y, oof_proba.argmax(axis=1))
    oof_f1 = f1_score(y, oof_proba.argmax(axis=1), average="macro")
    print(f"\nOOF accuracy={oof_acc:.6f}  f1_macro={oof_f1:.6f}")

    np.save(MODELS / "dart_oof_proba.npy", oof_proba)
    np.save(MODELS / "dart_test_proba.npy", test_proba)

    (MODELS / "dart_metrics.json").write_text(
        json.dumps({"oof_accuracy": oof_acc, "oof_f1_macro": oof_f1, "folds": fold_rows}, indent=2))

    pred_labels = target_encoder.inverse_transform(test_proba.argmax(axis=1))
    sub = sample.copy()
    sub.iloc[:, 1] = pred_labels
    sub_path = SUBMISSIONS / "dart_submission.csv"
    sub.to_csv(sub_path, index=False)
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                    "-f", str(sub_path), "-m", "DART LGB feateng 1200est 10fold"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
