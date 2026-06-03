#!/usr/bin/env python3
"""LGB v6 — MCC-safe. Columns: id,alpha,delta,u,g,r,i,z,redshift,spectral_type,galaxy_population,class"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb

DATA_DIR = Path.home() / "kaggle" / "s6e6" / "data"
OUT_DIR  = Path.home() / "kaggle" / "s6e6" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 10
TARGET = "class"

PARAMS = {
    "num_leaves": 98,
    "learning_rate": 0.01469,
    "n_estimators": 2000,
    "max_depth": 10,
    "min_child_samples": 14,
    "subsample": 0.842,
    "colsample_bytree": 0.664,
    "reg_alpha": 0.244,
    "reg_lambda": 2.476,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    ra_rad  = np.deg2rad(df["alpha"])
    dec_rad = np.deg2rad(df["delta"])
    df["galactic_l"] = np.arctan2(
        np.sin(ra_rad - np.deg2rad(282.25)) * np.cos(dec_rad),
        np.cos(ra_rad - np.deg2rad(282.25)) * np.cos(dec_rad) * np.sin(np.deg2rad(62.6))
        - np.sin(dec_rad) * np.cos(np.deg2rad(62.6)),
    )
    df["galactic_b"] = np.arcsin(
        np.sin(dec_rad) * np.sin(np.deg2rad(62.6))
        + np.cos(dec_rad) * np.cos(np.deg2rad(62.6)) * np.cos(ra_rad - np.deg2rad(282.25))
    )
    df["ra_dec_ratio"]  = df["alpha"] / (df["delta"].abs() + 1e-6)
    df["sky_area"]      = np.cos(dec_rad)
    mags = df[["u", "g", "r", "i", "z"]]
    df["mag_range"]    = mags.max(axis=1) - mags.min(axis=1)
    df["mag_mean"]     = mags.mean(axis=1)
    df["mag_std"]      = mags.std(axis=1)
    df["mag_skew"]     = mags.skew(axis=1)
    df["u_g"]          = df["u"] - df["g"]
    df["g_r"]          = df["g"] - df["r"]
    df["r_i"]          = df["r"] - df["i"]
    df["i_z"]          = df["i"] - df["z"]
    df["u_r"]          = df["u"] - df["r"]
    df["g_i"]          = df["g"] - df["i"]
    df["g_z"]          = df["g"] - df["z"]
    df["u_z"]          = df["u"] - df["z"]
    df["r_z"]          = df["r"] - df["z"]
    df["u_i"]          = df["u"] - df["i"]
    df["g_r_ratio"]    = df["g_r"] / (df["r_i"].abs() + 1e-6)
    df["color_slope"]  = (df["z"] - df["u"]) / 4.0
    df["redshift_log"] = np.log1p(df["redshift"].clip(0))
    df["redshift_sq"]  = df["redshift"] ** 2
    df["redshift_bin"] = pd.cut(df["redshift"], bins=20, labels=False)
    df["ra_sector"]    = (df["alpha"] // 30).astype(int)
    df["dec_band"]     = pd.cut(df["delta"], bins=12, labels=False)
    return df


def main() -> int:
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    for col in ["spectral_type", "galaxy_population"]:
        if col in train.columns:
            combined = pd.concat([train[col], test[col]], axis=0)
            le_col = LabelEncoder()
            le_col.fit(combined.fillna("unknown"))
            train[col] = le_col.transform(train[col].fillna("unknown"))
            test[col]  = le_col.transform(test[col].fillna("unknown"))

    train = add_features(train)
    test  = add_features(test)

    drop_cols = [TARGET, "id"]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    cat_cols  = [c for c in ["spectral_type", "galaxy_population"] if c in feat_cols]

    le = LabelEncoder()
    y  = le.fit_transform(train[TARGET])
    classes = le.classes_
    print(f"Features: {len(feat_cols)}  Cat: {cat_cols}  Classes: {classes}", flush=True)

    X_train = train[feat_cols]
    X_test  = test[[c for c in feat_cols if c in test.columns]]

    for col in feat_cols:
        if col not in X_test.columns:
            X_test = X_test.copy()
            X_test[col] = 0
    X_test = X_test[feat_cols]

    oof   = np.zeros((len(train), len(classes)), dtype=np.float32)
    ptest = np.zeros((len(test),  len(classes)), dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"Fold {fold+1}/{N_FOLDS}...", flush=True)
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X_train.iloc[tr_idx], y[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y[val_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        )
        oof[val_idx]  = model.predict_proba(X_train.iloc[val_idx])
        ptest        += model.predict_proba(X_test) / N_FOLDS

    acc  = accuracy_score(y, oof.argmax(axis=1))
    loss = log_loss(y, oof)
    print(f"OOF accuracy={acc:.6f}  logloss={loss:.8f}", flush=True)

    np.save(OUT_DIR / "lgbv6_mcc_oof_proba.npy", oof)
    np.save(OUT_DIR / "lgbv6_mcc_test_proba.npy", ptest)
    with open(OUT_DIR / "lgbv6_mcc_metrics.json", "w") as f:
        json.dump({"oof_accuracy": float(acc), "oof_logloss": float(loss),
                   "n_features": len(feat_cols)}, f)
    print("Saved. Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
