#!/usr/bin/env python3
"""LightGBM + feature engineering for S6E6 Stellar Class.

Adds astrophysics color indices (u-g, g-r, r-i, i-z, u-r, g-z),
log-redshift, and galactic coordinates to close the OOF/LB gap.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
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
    # Color indices (magnitude differences — standard astrophysics)
    df["u_minus_g"] = df["u"] - df["g"]
    df["g_minus_r"] = df["g"] - df["r"]
    df["r_minus_i"] = df["r"] - df["i"]
    df["i_minus_z"] = df["i"] - df["z"]
    df["u_minus_r"] = df["u"] - df["r"]
    df["g_minus_z"] = df["g"] - df["z"]
    df["u_minus_z"] = df["u"] - df["z"]
    df["g_minus_i"] = df["g"] - df["i"]
    # Log redshift (highly skewed; stars cluster near 0 so clip at 1e-4)
    df["log_redshift"] = np.log1p(df["redshift"].clip(lower=0))
    # Redshift squared (QSO separation)
    df["redshift_sq"] = df["redshift"] ** 2
    # Total optical flux proxy
    df["mean_mag"] = df[["u", "g", "r", "i", "z"]].mean(axis=1)
    df["mag_std"] = df[["u", "g", "r", "i", "z"]].std(axis=1)
    df["mag_range"] = df[["u", "g", "r", "i", "z"]].max(axis=1) - df[["u", "g", "r", "i", "z"]].min(axis=1)
    # Sky position — galactic-ish proxy
    df["alpha_rad"] = np.radians(df["alpha"])
    df["delta_rad"] = np.radians(df["delta"])
    df["cos_alpha"] = np.cos(df["alpha_rad"])
    df["sin_alpha"] = np.sin(df["alpha_rad"])
    df["cos_delta"] = np.cos(df["delta_rad"])
    return df


def preprocess(train_x: pd.DataFrame, test_x: pd.DataFrame):
    train_x = add_features(train_x)
    test_x = add_features(test_x)
    cat_cols = []
    for col in train_x.columns:
        if train_x[col].dtype == "object" or str(train_x[col].dtype).startswith("category"):
            cat_cols.append(col)
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_x[col], test_x[col]], axis=0).astype(str).fillna("__MISSING__")
        le.fit(combined)
        train_x[col] = le.transform(train_x[col].astype(str).fillna("__MISSING__"))
        test_x[col] = le.transform(test_x[col].astype(str).fillna("__MISSING__"))
    for col in train_x.columns:
        if train_x[col].isna().any() or test_x[col].isna().any():
            median = train_x[col].median()
            train_x[col] = train_x[col].fillna(0 if pd.isna(median) else median)
            test_x[col] = test_x[col].fillna(0 if pd.isna(median) else median)
    # Drop raw angle columns (kept derived trig versions)
    drop_cols = ["alpha_rad", "delta_rad"]
    train_x = train_x.drop(columns=[c for c in drop_cols if c in train_x.columns])
    test_x = test_x.drop(columns=[c for c in drop_cols if c in test_x.columns])
    return train_x, test_x, cat_cols


def main() -> int:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise SystemExit(f"lightgbm required: {exc}")

    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    id_col = "id"
    target = "class"
    feature_cols = [c for c in train.columns if c not in (target, id_col)]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train[target].astype(str))
    classes = list(target_encoder.classes_)
    n_classes = len(classes)

    x_train, x_test, cat_cols = preprocess(train[feature_cols], test[feature_cols])
    print(f"Features: {x_train.shape[1]} (was {len(feature_cols)})")

    params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": "multiclass",
        "num_class": n_classes,
    }

    oof_proba = np.zeros((len(train), n_classes))
    test_proba = np.zeros((len(test), n_classes))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), 1):
        model = LGBMClassifier(**params)
        model.fit(
            x_train.iloc[tr_idx], y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[],
        )
        va_proba = model.predict_proba(x_train.iloc[va_idx])
        te_proba = model.predict_proba(x_test)
        oof_proba[va_idx] = va_proba
        test_proba += te_proba / skf.get_n_splits()
        va_pred = va_proba.argmax(axis=1)
        acc = accuracy_score(y[va_idx], va_pred)
        f1m = f1_score(y[va_idx], va_pred, average="macro")
        fold_rows.append({"fold": fold, "accuracy": acc, "f1_macro": f1m})
        print(f"fold={fold:2d} accuracy={acc:.6f} f1_macro={f1m:.6f}")

    oof_pred = oof_proba.argmax(axis=1)
    oof_acc = accuracy_score(y, oof_pred)
    oof_f1 = f1_score(y, oof_pred, average="macro")
    print(f"\nOOF accuracy={oof_acc:.6f}")
    print(f"OOF f1_macro={oof_f1:.6f}")

    pred_labels = target_encoder.inverse_transform(test_proba.argmax(axis=1))
    submission = sample.copy()
    submission.iloc[:, 1] = pred_labels
    sub_path = SUBMISSIONS / "feateng_lgb_submission.csv"
    submission.to_csv(sub_path, index=False)

    metrics = {
        "model": "feateng_lgb",
        "folds": 10,
        "features": x_train.shape[1],
        "fold_results": fold_rows,
        "oof_accuracy": oof_acc,
        "oof_f1_macro": oof_f1,
        "submission": str(sub_path),
    }
    (MODELS / "feateng_metrics.json").write_text(json.dumps(metrics, indent=2))

    cmd = ["kaggle", "competitions", "submit", "-c", COMP, "-f", str(sub_path), "-m", "feateng LGB 10fold color-indices"]
    print("Submitting:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"Submit failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
