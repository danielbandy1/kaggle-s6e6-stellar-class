#!/usr/bin/env python3
"""LightGBM baseline for Kaggle Playground S6E6: Predicting Stellar Class.

Loads train/test/sample_submission, performs minimal preprocessing, trains a
5-fold StratifiedKFold LightGBM classifier, saves OOF/test predictions, writes a
submission, and optionally submits it through the Kaggle CLI.
"""

from __future__ import annotations

import argparse
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


def infer_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    candidates = [c for c in train.columns if c not in test.columns and c.lower() != "id"]
    if not candidates:
        raise ValueError("Could not infer target column. Expected a train column absent from test.csv.")
    return candidates[-1]


def preprocess(train_x: pd.DataFrame, test_x: pd.DataFrame):
    train_x = train_x.copy()
    test_x = test_x.copy()
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
            if pd.isna(median):
                median = 0
            train_x[col] = train_x[col].fillna(median)
            test_x[col] = test_x[col].fillna(median)

    return train_x, test_x, cat_cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true", help="Do not call kaggle competitions submit")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise SystemExit(f"lightgbm is required. Install it with: pip install lightgbm\nImport error: {exc}")

    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train_path = DATA / "train.csv"
    test_path = DATA / "test.csv"
    sample_path = DATA / "sample_submission.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing train/test CSVs under {DATA}. Run Kaggle download first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None

    id_col = "id" if "id" in train.columns and "id" in test.columns else None
    target = infer_target(train, test)
    feature_cols = [c for c in train.columns if c != target and c != id_col]

    y_raw = train[target]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw.astype(str))
    classes = list(target_encoder.classes_)
    n_classes = len(classes)

    x_train, x_test, cat_cols = preprocess(train[feature_cols], test[feature_cols])

    params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if n_classes > 2:
        params.update({"objective": "multiclass", "num_class": n_classes})
        oof_proba = np.zeros((len(train), n_classes))
        test_proba = np.zeros((len(test), n_classes))
    else:
        params.update({"objective": "binary"})
        oof_proba = np.zeros((len(train), 2))
        test_proba = np.zeros((len(test), 2))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), 1):
        model = LGBMClassifier(**params)
        model.fit(
            x_train.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss" if n_classes > 2 else "binary_logloss",
        )
        va_proba = model.predict_proba(x_train.iloc[va_idx])
        te_proba = model.predict_proba(x_test)
        if n_classes == 2 and va_proba.shape[1] == 1:
            va_proba = np.c_[1 - va_proba[:, 0], va_proba[:, 0]]
            te_proba = np.c_[1 - te_proba[:, 0], te_proba[:, 0]]
        oof_proba[va_idx] = va_proba
        test_proba += te_proba / args.folds
        va_pred = va_proba.argmax(axis=1)
        acc = accuracy_score(y[va_idx], va_pred)
        f1m = f1_score(y[va_idx], va_pred, average="macro")
        fold_rows.append({"fold": fold, "accuracy": acc, "f1_macro": f1m})
        print(f"fold={fold} accuracy={acc:.6f} f1_macro={f1m:.6f}")

    oof_pred = oof_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)
    oof_acc = accuracy_score(y, oof_pred)
    oof_f1 = f1_score(y, oof_pred, average="macro")
    print(f"OOF accuracy={oof_acc:.6f}")
    print(f"OOF f1_macro={oof_f1:.6f}")

    pred_labels = target_encoder.inverse_transform(test_pred)
    oof_labels = target_encoder.inverse_transform(oof_pred)

    oof_df = pd.DataFrame({target: y_raw, f"{target}_pred": oof_labels})
    if id_col:
        oof_df.insert(0, id_col, train[id_col].values)
    for i, cls in enumerate(classes):
        oof_df[f"proba_{cls}"] = oof_proba[:, i]
    oof_df.to_csv(MODELS / "oof_predictions.csv", index=False)

    test_pred_df = pd.DataFrame()
    if id_col:
        test_pred_df[id_col] = test[id_col].values
    for i, cls in enumerate(classes):
        test_pred_df[f"proba_{cls}"] = test_proba[:, i]
    test_pred_df[f"{target}_pred"] = pred_labels
    test_pred_df.to_csv(MODELS / "test_predictions.csv", index=False)

    if sample is not None and len(sample.columns) >= 2:
        submission = sample.copy()
        submission.iloc[:, 1] = pred_labels
    else:
        sub_id = test[id_col].values if id_col else np.arange(len(test))
        submission = pd.DataFrame({"id": sub_id, target: pred_labels})
    sub_path = SUBMISSIONS / "baseline_lgb_submission.csv"
    submission.to_csv(sub_path, index=False)

    metrics = {
        "target": target,
        "classes": classes,
        "features": feature_cols,
        "categorical_features": cat_cols,
        "folds": fold_rows,
        "oof_accuracy": oof_acc,
        "oof_f1_macro": oof_f1,
        "submission": str(sub_path),
    }
    (MODELS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.no_submit:
        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMP,
            "-f",
            str(sub_path),
            "-m",
            "baseline LGB",
        ]
        print("Submitting:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            print(f"Kaggle submit failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
