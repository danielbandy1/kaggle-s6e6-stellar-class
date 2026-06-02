#!/usr/bin/env python3
"""Optuna LightGBM tuning for S6E6. Intended for MCC SLURM execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path.home() / "kaggle" / "s6e6"
DATA = ROOT / "data"
MODELS = ROOT / "models"


def infer_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    candidates = [c for c in train.columns if c not in test.columns and c.lower() != "id"]
    if not candidates:
        raise ValueError("Could not infer target column")
    return candidates[-1]


def preprocess(train_x: pd.DataFrame, test_x: pd.DataFrame):
    train_x = train_x.copy()
    test_x = test_x.copy()
    for col in train_x.columns:
        if train_x[col].dtype == "object" or str(train_x[col].dtype).startswith("category"):
            le = LabelEncoder()
            combined = pd.concat([train_x[col], test_x[col]], axis=0).astype(str).fillna("__MISSING__")
            le.fit(combined)
            train_x[col] = le.transform(train_x[col].astype(str).fillna("__MISSING__"))
            test_x[col] = le.transform(test_x[col].astype(str).fillna("__MISSING__"))
        if train_x[col].isna().any() or test_x[col].isna().any():
            median = train_x[col].median()
            if pd.isna(median):
                median = 0
            train_x[col] = train_x[col].fillna(median)
            test_x[col] = test_x[col].fillna(median)
    return train_x, test_x


def main() -> int:
    import optuna
    from lightgbm import LGBMClassifier

    MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    id_col = "id" if "id" in train.columns and "id" in test.columns else None
    target = infer_target(train, test)
    features = [c for c in train.columns if c != target and c != id_col]
    x_train, _ = preprocess(train[features], test[features])
    y = LabelEncoder().fit_transform(train[target].astype(str))
    n_classes = len(np.unique(y))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "random_state": 42,
            "n_jobs": 8,
            "verbosity": -1,
        }
        if n_classes > 2:
            params.update({"objective": "multiclass", "num_class": n_classes})
        else:
            params.update({"objective": "binary"})

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in skf.split(x_train, y):
            model = LGBMClassifier(**params)
            model.fit(x_train.iloc[tr_idx], y[tr_idx])
            pred = model.predict(x_train.iloc[va_idx])
            scores.append(f1_score(y[va_idx], pred, average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    result = {
        "best_value_f1_macro": study.best_value,
        "best_params": study.best_params,
        "target": target,
        "n_classes": n_classes,
        "features": features,
        "n_trials": 100,
    }
    (MODELS / "best_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
