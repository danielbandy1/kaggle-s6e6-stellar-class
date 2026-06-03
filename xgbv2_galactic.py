#!/usr/bin/env python3
"""XGBoost v2 — galactic coordinates + photometric SED features.

Same feature engineering as LGBv4 (61 features) but trained with XGBoost.
Provides ensemble diversity over the LGB variants.
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
from xgboost import XGBClassifier

ROOT = Path("/home/daniel/kaggle/s6e6")
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
COMP = "playground-series-s6e6"

WAVELENGTHS = np.array([354.3, 477.0, 623.1, 762.5, 913.4], dtype=np.float32)
LOG_WAVE = np.log(WAVELENGTHS)
LOG_WAVE_CENTERED = LOG_WAVE - LOG_WAVE.mean()


def equatorial_to_galactic(alpha_deg: pd.Series, delta_deg: pd.Series):
    alpha_gp = np.radians(192.85948)
    delta_gp = np.radians(27.12825)
    l_ncp    = np.radians(122.93192)

    alpha = np.radians(alpha_deg)
    delta = np.radians(delta_deg)

    sin_b = (np.sin(delta_gp) * np.sin(delta)
             + np.cos(delta_gp) * np.cos(delta) * np.cos(alpha - alpha_gp))
    b = np.arcsin(sin_b.clip(-1, 1))

    y = np.cos(delta) * np.sin(alpha - alpha_gp)
    x = (np.sin(delta) * np.cos(delta_gp)
         - np.cos(delta) * np.sin(delta_gp) * np.cos(alpha - alpha_gp))
    l = l_ncp - np.arctan2(y, x)
    l = l % (2 * np.pi)

    return pd.Series(np.degrees(l), index=alpha_deg.index), pd.Series(np.degrees(b), index=alpha_deg.index)


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
    df["r_minus_z"] = df["r"] - df["i"]
    df["u_minus_i"] = df["u"] - df["i"]

    df["color_curve_ugr"] = df["u_minus_g"] - df["g_minus_r"]
    df["color_curve_gri"] = df["g_minus_r"] - df["r_minus_i"]
    df["color_curve_riz"] = df["r_minus_i"] - df["i_minus_z"]
    df["spectral_slope"]  = (df["u"] - df["z"]) / 4.0

    mags_array = df[["u", "g", "r", "i", "z"]].values.astype(np.float64)
    sed_slopes = np.polyfit(LOG_WAVE_CENTERED, mags_array.T, 1)
    df["sed_slope"]     = sed_slopes[0]
    df["sed_amplitude"] = sed_slopes[1]
    df["sed_residual"]  = mags_array.mean(axis=1) - sed_slopes[1]

    df["locus_diag1"] = df["u_minus_g"] - 0.88 * df["g_minus_r"]
    df["locus_diag2"] = df["g_minus_r"] - 0.35 * df["r_minus_i"]

    df["log_redshift"]  = np.log1p(df["redshift"].clip(lower=0))
    df["redshift_sq"]   = df["redshift"] ** 2
    df["redshift_cube"] = df["redshift"] ** 3
    df["is_high_z"]     = (df["redshift"] > 0.5).astype(np.float32)
    df["is_zero_z"]     = (df["redshift"].abs() < 0.01).astype(np.float32)
    df["is_star_z"]     = (df["redshift"].abs() < 0.002).astype(np.float32)
    df["signed_log_z"]  = np.sign(df["redshift"]) * np.log1p(df["redshift"].abs())

    df["z_x_gmr"] = df["redshift"] * df["g_minus_r"]
    df["z_x_umg"] = df["redshift"] * df["u_minus_g"]
    df["z_x_rmi"] = df["redshift"] * df["r_minus_i"]
    df["z_x_ugz"] = df["redshift"] * df["u_minus_z"]

    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"] = mags.mean(axis=1)
    df["mag_std"]  = mags.std(axis=1)
    df["mag_range"] = mags.max(axis=1) - mags.min(axis=1)
    df["mag_skew"]  = mags.skew(axis=1)

    l_deg, b_deg = equatorial_to_galactic(df["alpha"], df["delta"])
    df["galactic_l"]     = l_deg
    df["galactic_b"]     = b_deg
    df["galactic_b_abs"] = b_deg.abs()
    df["sin_b"]          = np.sin(np.radians(b_deg))
    df["cos_b"]          = np.cos(np.radians(b_deg))
    df["cos_l"]          = np.cos(np.radians(l_deg))
    df["sin_l"]          = np.sin(np.radians(l_deg))
    df["near_plane"]     = (b_deg.abs() < 10).astype(np.float32)
    df["high_lat"]       = (b_deg.abs() > 60).astype(np.float32)
    df["z_x_b"]          = df["redshift"] * df["galactic_b_abs"]
    df["z_x_sinb"]       = df["redshift"] * df["sin_b"]

    df["cos_alpha"]  = np.cos(np.radians(df["alpha"]))
    df["sin_alpha"]  = np.sin(np.radians(df["alpha"]))
    df["cos_delta"]  = np.cos(np.radians(df["delta"]))
    df["sin_delta"]  = np.sin(np.radians(df["delta"]))
    df["cos_2alpha"] = np.cos(2 * np.radians(df["alpha"]))
    df["sin_2alpha"] = np.sin(2 * np.radians(df["alpha"]))

    return df


def preprocess(train_x, test_x):
    train_x = add_features(train_x)
    test_x  = add_features(test_x)

    cat_cols = [c for c in train_x.columns if train_x[c].dtype == "object"]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_x[col], test_x[col]]).astype(str).fillna("__MISSING__")
        le.fit(combined)
        train_x[col] = le.transform(train_x[col].astype(str).fillna("__MISSING__"))
        test_x[col]  = le.transform(test_x[col].astype(str).fillna("__MISSING__"))

    for col in train_x.columns:
        if train_x[col].isna().any():
            m = train_x[col].median()
            train_x[col] = train_x[col].fillna(0 if pd.isna(m) else m)
            test_x[col]  = test_x[col].fillna(0 if pd.isna(m) else m)

    return train_x, test_x


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train  = pd.read_csv(DATA / "train.csv")
    test   = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train["class"].astype(str))
    classes = list(target_encoder.classes_)
    n_classes = len(classes)
    print(f"Classes: {classes}")

    train_x = train.drop(columns=[c for c in ["id", "class"] if c in train.columns])
    test_x  = test.drop(columns=[c for c in ["id"] if c in test.columns])
    train_x, test_x = preprocess(train_x, test_x)
    print(f"Feature count: {len(train_x.columns)}")

    params = dict(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="merror",
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=100,
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_proba  = np.zeros((len(train), n_classes))
    test_proba = np.zeros((len(test),  n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_x, y)):
        X_tr, X_va = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        oof_proba[va_idx] = model.predict_proba(X_va)
        test_proba += model.predict_proba(test_x) / skf.get_n_splits()

        fold_acc = accuracy_score(y_va, oof_proba[va_idx].argmax(axis=1))
        print(f"Fold {fold+1:2d}  OOF acc={fold_acc:.6f}  trees={model.best_iteration}", flush=True)

    oof_acc = accuracy_score(y, oof_proba.argmax(axis=1))
    oof_f1  = f1_score(y, oof_proba.argmax(axis=1), average="macro")
    print(f"\nFull OOF  acc={oof_acc:.6f}  f1_macro={oof_f1:.6f}")

    np.save(MODELS / "xgbv2_oof_proba.npy",  oof_proba)
    np.save(MODELS / "xgbv2_test_proba.npy", test_proba)

    pred_labels = target_encoder.inverse_transform(test_proba.argmax(axis=1))
    sub = sample.copy()
    sub.iloc[:, 1] = pred_labels
    sub_path = SUBMISSIONS / f"xgbv2_galactic_oof{oof_acc:.5f}.csv"
    sub.to_csv(sub_path, index=False)

    msg = f"XGBv2 galactic+SED OOF={oof_acc:.5f} f1={oof_f1:.5f}"
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                    "-f", str(sub_path), "-m", msg], check=False)

    (MODELS / "xgbv2_metrics.json").write_text(json.dumps({
        "model": "xgbv2", "oof_accuracy": oof_acc, "oof_f1_macro": oof_f1,
        "n_features": len(train_x.columns),
    }, indent=2))
    print(f"\nDone. OOF acc={oof_acc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
