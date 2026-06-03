#!/usr/bin/env python3
"""LGB v5 - redshift transforms + color ratios + sky/locus features.

Built from the working v3 template, with ideas prioritized from the S6E6
discussion/notebook sweep and SDSS stellar-classification domain patterns:
  1. Redshift is the strongest spectroscopic separator: add signed log1p,
     absolute/sqrt/inverse transforms, fixed bins, train-fit quantile bins,
     and QuantileTransformer projections.
  2. Color space carries the photometric locus: add every pairwise color,
     adjacent 4-color ratios, UV excess, red/blue slopes, and color curvature.
  3. Sky position can capture survey footprint artifacts: sinusoidal RA/Dec
     harmonics, unit-sphere coordinates, and coarse sky bins.
  4. No photometric errors are provided, so add noise proxies from SED residuals,
     color jump roughness, flux concentration, and outlier band residuals.
  5. Add class-locus heuristics for STAR/QSO/GALAXY separation without target
     leakage: near-zero-z stellar smoothness, high-z blue QSO score, and red
     sequence galaxy distance.

This script intentionally does not submit to Kaggle. It saves:
  models/lgbv5_oof_proba.npy
  models/lgbv5_test_proba.npy
  models/lgbv5_metrics.json
  submissions/lgbv5_submission.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

ROOT = Path("/home/daniel/kaggle/s6e6")
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

EPS = 1e-6
BANDS = ["u", "g", "r", "i", "z"]
WAVELENGTHS = np.array([354.3, 477.0, 623.1, 762.5, 913.4], dtype=np.float32)
LOG_WAVE_CENTERED = np.log(WAVELENGTHS) - np.log(WAVELENGTHS).mean()
LOG_WAVE_DENOM = float(np.sum(LOG_WAVE_CENTERED ** 2))


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan).fillna(0) + EPS)


def optional_col(df: pd.DataFrame, *names: str) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        match = lower_to_col.get(name.lower())
        if match is not None:
            return match
    return None


def angle_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    ra_col = "alpha" if "alpha" in df.columns else ("ra" if "ra" in df.columns else None)
    dec_col = "delta" if "delta" in df.columns else ("dec" if "dec" in df.columns else None)
    return ra_col, dec_col


def equatorial_to_galactic(ra_deg: pd.Series, dec_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert J2000 RA/Dec to Galactic longitude/latitude in degrees."""
    alpha_gp = np.radians(192.85948)
    delta_gp = np.radians(27.12825)
    l_ncp = np.radians(122.93192)

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    sin_b = (
        np.sin(delta_gp) * np.sin(dec)
        + np.cos(delta_gp) * np.cos(dec) * np.cos(ra - alpha_gp)
    )
    b = np.arcsin(np.clip(sin_b, -1, 1))

    y = np.cos(dec) * np.sin(ra - alpha_gp)
    x = (
        np.sin(dec) * np.cos(delta_gp)
        - np.cos(dec) * np.sin(delta_gp) * np.cos(ra - alpha_gp)
    )
    l = (l_ncp - np.arctan2(y, x)) % (2 * np.pi)
    return pd.Series(np.degrees(l), index=ra_deg.index), pd.Series(np.degrees(b), index=ra_deg.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Core and full pairwise color indices ---
    for left_i, left in enumerate(BANDS):
        for right in BANDS[left_i + 1:]:
            df[f"{left}_minus_{right}"] = df[left] - df[right]

    df["color_curve_ugr"] = df["u_minus_g"] - df["g_minus_r"]
    df["color_curve_gri"] = df["g_minus_r"] - df["r_minus_i"]
    df["color_curve_riz"] = df["r_minus_i"] - df["i_minus_z"]
    df["spectral_slope"] = df["u_minus_z"] / 4.0
    df["blue_slope"] = (df["u"] + df["g"]) - (df["i"] + df["z"])
    df["red_slope"] = (df["r"] + df["i"] + df["z"]) / 3.0 - (df["u"] + df["g"]) / 2.0
    df["uv_excess"] = df["u_minus_g"] - 0.5 * df["g_minus_r"]

    # Adjacent 4-color ratios and ratio-of-ratios. These are noisy but often
    # useful for tree splits near stellar/QSO color-locus boundaries.
    adjacent_colors = ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]
    for i, col_a in enumerate(adjacent_colors):
        for col_b in adjacent_colors[i + 1:]:
            df[f"{col_a}_div_{col_b}"] = safe_ratio(df[col_a], df[col_b])
            df[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]

    df["ug_gr_ri_ratio"] = safe_ratio(df["u_minus_g"] + df["g_minus_r"], df["r_minus_i"] + EPS)
    df["gr_ri_iz_ratio"] = safe_ratio(df["g_minus_r"] + df["r_minus_i"], df["i_minus_z"] + EPS)
    df["u_to_z_color_ratio"] = safe_ratio(df["u_minus_z"], df["g_minus_i"].abs() + EPS)
    df["blue_to_red_color_ratio"] = safe_ratio(df["u_minus_r"], df["r_minus_z"].abs() + EPS)

    # --- Redshift transforms and spectroscopic regimes ---
    z = df["redshift"]
    abs_z = z.abs()
    df["redshift_abs"] = abs_z
    df["redshift_signed_log1p"] = np.sign(z) * np.log1p(abs_z)
    df["redshift_log1p_pos"] = np.log1p(z.clip(lower=0))
    df["redshift_log1p_abs"] = np.log1p(abs_z)
    df["redshift_sqrt_abs"] = np.sqrt(abs_z)
    df["redshift_inv1p_abs"] = 1.0 / (1.0 + abs_z)
    df["redshift_sq"] = z ** 2
    df["redshift_cube"] = z ** 3
    df["redshift_neg"] = (z < -0.001).astype(np.float32)
    df["redshift_near_zero_0005"] = (abs_z < 0.0005).astype(np.float32)
    df["redshift_near_zero_002"] = (abs_z < 0.002).astype(np.float32)
    df["redshift_star_window"] = (abs_z < 0.02).astype(np.float32)
    df["redshift_galaxy_window"] = ((z >= 0.02) & (z < 0.8)).astype(np.float32)
    df["redshift_qso_window"] = (z >= 0.8).astype(np.float32)
    z_bins = [-np.inf, -0.01, 0.0005, 0.002, 0.01, 0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.5, np.inf]
    df["redshift_bin_fixed"] = pd.cut(z, bins=z_bins, labels=False, include_lowest=True).fillna(-1).astype(np.int16) + 1
    df["redshift_bin_centered"] = df["redshift_bin_fixed"].astype(np.float32) - 6.0

    # --- Redshift x color interactions ---
    for col in ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z", "u_minus_r", "u_minus_z"]:
        df[f"z_x_{col}"] = z * df[col]
        df[f"logz_x_{col}"] = df["redshift_signed_log1p"] * df[col]
        df[f"zbin_x_{col}"] = df["redshift_bin_centered"] * df[col]

    # --- Magnitude statistics and photometric noise proxies ---
    mags = df[BANDS]
    mag_values = mags.to_numpy(dtype=np.float64)
    df["mean_mag"] = mags.mean(axis=1)
    df["median_mag"] = mags.median(axis=1)
    df["mag_std"] = mags.std(axis=1)
    df["mag_range"] = mags.max(axis=1) - mags.min(axis=1)
    df["mag_min"] = mags.min(axis=1)
    df["mag_max"] = mags.max(axis=1)
    df["mag_skew"] = mags.apply(lambda row: pd.Series(row).skew(), axis=1)
    for col in ["mean_mag", "mag_std", "mag_range", "mag_skew"]:
        df[f"zbin_x_{col}"] = df["redshift_bin_centered"] * df[col]

    color_mat = df[adjacent_colors].to_numpy(dtype=np.float64)
    df["color_adjacent_std"] = np.std(color_mat, axis=1)
    df["color_adjacent_range"] = np.max(color_mat, axis=1) - np.min(color_mat, axis=1)
    df["color_abs_sum"] = np.abs(color_mat).sum(axis=1)
    df["color_roughness"] = (
        np.abs(df["color_curve_ugr"]) + np.abs(df["color_curve_gri"]) + np.abs(df["color_curve_riz"])
    )

    # SED linear fit in magnitude space; residuals are useful photometric-shape/noise proxies.
    sed_slope = ((mag_values - mag_values.mean(axis=1, keepdims=True)) @ LOG_WAVE_CENTERED) / LOG_WAVE_DENOM
    sed_amp = mag_values.mean(axis=1)
    fitted = sed_amp[:, None] + sed_slope[:, None] * LOG_WAVE_CENTERED[None, :]
    resid = mag_values - fitted
    df["sed_slope"] = sed_slope
    df["sed_amplitude"] = sed_amp
    df["sed_resid_rms"] = np.sqrt(np.mean(resid ** 2, axis=1))
    df["sed_resid_max_abs"] = np.max(np.abs(resid), axis=1)
    for idx, band in enumerate(BANDS):
        df[f"sed_resid_{band}"] = resid[:, idx]

    # Convert magnitudes to relative flux proxies. Absolute scale is arbitrary,
    # but normalized concentrations and flux ratios capture SED shape differently
    # from magnitude differences.
    flux = np.power(10.0, -0.4 * np.clip(mag_values, -50, 50))
    flux_sum = flux.sum(axis=1) + EPS
    for idx, band in enumerate(BANDS):
        df[f"flux_frac_{band}"] = flux[:, idx] / flux_sum
    df["flux_sum_log"] = np.log1p(flux_sum)
    df["flux_blue_frac"] = (flux[:, 0] + flux[:, 1]) / flux_sum
    df["flux_red_frac"] = (flux[:, 3] + flux[:, 4]) / flux_sum
    df["flux_concentration"] = flux.max(axis=1) / flux_sum
    df["morphology_proxy_blue_red_flux_ratio"] = safe_ratio(df["flux_blue_frac"], df["flux_red_frac"])
    df["morphology_proxy_color_compactness"] = safe_ratio(df["mag_range"], df["mean_mag"].abs() + EPS)

    # Optional SDSS morphology block. The S6E6 files currently expose only PSF-like
    # band magnitudes, but some SDSS mirrors include petroMag_* and fiberMag_*.
    # Add ratios when present and otherwise rely on the non-crashing proxies above.
    petro_fiber_cols: list[str] = []
    for band in BANDS:
        petro = optional_col(
            df,
            f"petroMag_{band}", f"petromag_{band}", f"petro_{band}",
            f"petrosian_{band}", f"petroMag_{band.upper()}", f"petroMag{band.upper()}",
        )
        fiber = optional_col(
            df,
            f"fiberMag_{band}", f"fibermag_{band}", f"fiber_{band}",
            f"fiberMag_{band.upper()}", f"fiberMag{band.upper()}",
        )
        if petro is None or fiber is None:
            continue
        ratio_col = f"petro_fiber_ratio_{band}"
        diff_col = f"petro_minus_fiber_{band}"
        df[ratio_col] = safe_ratio(df[petro], df[fiber])
        df[diff_col] = df[petro] - df[fiber]
        petro_fiber_cols.extend([ratio_col, diff_col])
    if petro_fiber_cols:
        morph = df[petro_fiber_cols]
        df["petro_fiber_ratio_mean"] = morph.mean(axis=1)
        df["petro_fiber_ratio_std"] = morph.std(axis=1)
        df["petro_fiber_ratio_range"] = morph.max(axis=1) - morph.min(axis=1)

    # --- Class-specific astrophysical locus heuristics (no target labels used) ---
    df["star_zeroz_smooth_score"] = df["redshift_inv1p_abs"] / (1.0 + df["color_roughness"])
    df["star_locus_dist"] = np.sqrt(
        (z / 0.01) ** 2
        + (df["color_curve_ugr"] / 0.45) ** 2
        + (df["color_curve_gri"] / 0.35) ** 2
        + (df["color_curve_riz"] / 0.35) ** 2
    )
    df["qso_blue_highz_score"] = sigmoid((z - 0.8) * 3.0) * sigmoid(-(df["u_minus_g"] - 0.6) * 2.0)
    df["qso_locus_dist"] = (
        np.maximum(0, 0.8 - z).abs()
        + np.abs(df["u_minus_g"] - 0.25)
        + 0.5 * np.abs(df["g_minus_r"] - 0.05)
    )
    df["galaxy_redseq_score"] = sigmoid((z - 0.03) * 8.0) * sigmoid((0.9 - z) * 4.0) * sigmoid((df["u_minus_z"] - 2.0))
    df["galaxy_locus_dist"] = (
        np.abs(df["u_minus_z"] - 3.0)
        + 0.7 * np.abs(df["g_minus_r"] - 1.0)
        + 0.5 * np.abs(z - 0.35)
    )

    # --- Survey sky-position features ---
    ra_col, dec_col = angle_columns(df)
    if ra_col is not None and dec_col is not None:
        ra_rad = np.radians(df[ra_col])
        dec_rad = np.radians(df[dec_col])
        for h in [1, 2, 3, 4]:
            df[f"sin_ra_{h}"] = np.sin(h * ra_rad)
            df[f"cos_ra_{h}"] = np.cos(h * ra_rad)
            df[f"sin_dec_{h}"] = np.sin(h * dec_rad)
            df[f"cos_dec_{h}"] = np.cos(h * dec_rad)

        for h in [3, 4]:
            df[f"sin_ra_{h}_x_sin_dec_{h}"] = df[f"sin_ra_{h}"] * df[f"sin_dec_{h}"]
            df[f"sin_ra_{h}_x_cos_dec_{h}"] = df[f"sin_ra_{h}"] * df[f"cos_dec_{h}"]
            df[f"cos_ra_{h}_x_sin_dec_{h}"] = df[f"cos_ra_{h}"] * df[f"sin_dec_{h}"]
            df[f"cos_ra_{h}_x_cos_dec_{h}"] = df[f"cos_ra_{h}"] * df[f"cos_dec_{h}"]

        df["sky_x"] = np.cos(dec_rad) * np.cos(ra_rad)
        df["sky_y"] = np.cos(dec_rad) * np.sin(ra_rad)
        df["sky_z"] = np.sin(dec_rad)
        df["ra_dec_interaction"] = df[ra_col] * df[dec_col]
        df["abs_dec"] = df[dec_col].abs()
        df["ra_sector_12"] = np.floor((df[ra_col] % 360) / 30.0).astype(np.int16)
        df["dec_band_12"] = pd.cut(df[dec_col], bins=np.linspace(-90, 90, 13), labels=False, include_lowest=True).fillna(-1).astype(np.int16) + 1

        gal_l, gal_b = equatorial_to_galactic(df[ra_col], df[dec_col])
        df["galactic_l"] = gal_l
        df["galactic_b"] = gal_b
        df["galactic_b_abs"] = gal_b.abs()
        df["sin_gal_l"] = np.sin(np.radians(gal_l))
        df["cos_gal_l"] = np.cos(np.radians(gal_l))
        df["sin_gal_b"] = np.sin(np.radians(gal_b))
        df["cos_gal_b"] = np.cos(np.radians(gal_b))
        df["near_galactic_plane"] = (gal_b.abs() < 10).astype(np.float32)
        df["high_galactic_lat"] = (gal_b.abs() > 60).astype(np.float32)
        df["z_x_abs_gal_b"] = z * df["galactic_b_abs"]

    # --- Optional survey identifiers if present in future versions/data mirrors ---
    for col in ["run", "camcol", "field"]:
        if col in df.columns:
            df[f"{col}_as_cat"] = df[col].fillna(-1).astype(np.int64)
            df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
    if "specobjid" in df.columns:
        spec = df["specobjid"].fillna(0)
        df["specobjid_log1p"] = np.log1p(np.abs(spec))
        df["specobjid_mod_1000"] = (spec.astype(np.int64) % 1000).astype(np.int16)

    return df


def add_train_fit_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add unsupervised train-fit features without using labels."""
    cat_cols: list[str] = []

    # Quantile bins are often stronger than fixed cuts for redshift because the
    # synthetic distribution is not uniform. Fit on train only, then apply to test.
    quantiles = np.unique(train_x["redshift"].quantile([0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1]).to_numpy())
    if len(quantiles) > 2:
        inner = quantiles[1:-1]
        train_x["redshift_qbin"] = np.digitize(train_x["redshift"], inner).astype(np.int16)
        test_x["redshift_qbin"] = np.digitize(test_x["redshift"], inner).astype(np.int16)
        cat_cols.append("redshift_qbin")

    for col in ["redshift_bin_fixed", "ra_sector_12", "dec_band_12", "specobjid_mod_1000"]:
        if col in train_x.columns:
            cat_cols.append(col)

    qt_sources = [
        "redshift",
        "redshift_signed_log1p",
        "redshift_abs",
        "u_minus_z",
        "u_minus_g",
        "g_minus_r",
        "mean_mag",
        "sed_slope",
        "sed_resid_rms",
    ]
    n_quantiles = min(1000, len(train_x))
    for col in qt_sources:
        if col not in train_x.columns:
            continue
        uniform_qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform", random_state=42)
        normal_qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=42)
        train_x[f"{col}_qt_uniform"] = uniform_qt.fit_transform(train_x[[col]]).ravel().astype(np.float32)
        test_x[f"{col}_qt_uniform"] = uniform_qt.transform(test_x[[col]]).ravel().astype(np.float32)
        train_x[f"{col}_qt_normal"] = normal_qt.fit_transform(train_x[[col]]).ravel().astype(np.float32)
        test_x[f"{col}_qt_normal"] = normal_qt.transform(test_x[[col]]).ravel().astype(np.float32)

    return train_x, test_x, cat_cols


def preprocess(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = add_features(train_x)
    test_x = add_features(test_x)
    train_x, test_x, cat_cols = add_train_fit_features(train_x, test_x)

    # Native LightGBM categorical handling for provided synthetic categories and
    # optional survey IDs. Keep the explicit cat_cols list tight to avoid turning
    # all integer-valued continuous transforms into categoricals.
    for col in ["spectral_type", "galaxy_population", "run_as_cat", "camcol_as_cat", "field_as_cat"]:
        if col in train_x.columns:
            le = LabelEncoder()
            combined = pd.concat([train_x[col], test_x[col]]).astype(str).fillna("__MISSING__")
            le.fit(combined)
            train_x[col] = le.transform(train_x[col].astype(str).fillna("__MISSING__")).astype(np.int32)
            test_x[col] = le.transform(test_x[col].astype(str).fillna("__MISSING__")).astype(np.int32)
            cat_cols.append(col)

    cat_cols = sorted(set(c for c in cat_cols if c in train_x.columns))

    for col in train_x.columns:
        if train_x[col].isna().any() or test_x[col].isna().any():
            if col in cat_cols:
                train_x[col] = train_x[col].fillna(0).astype(np.int32)
                test_x[col] = test_x[col].fillna(0).astype(np.int32)
            else:
                median = train_x[col].median()
                fill_value = 0 if pd.isna(median) else median
                train_x[col] = train_x[col].fillna(fill_value)
                test_x[col] = test_x[col].fillna(fill_value)

    # LightGBM is happier with finite values after aggressive ratio features.
    train_x = train_x.replace([np.inf, -np.inf], 0)
    test_x = test_x.replace([np.inf, -np.inf], 0)
    return train_x, test_x, cat_cols


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train["class"].astype(str))
    classes = list(target_encoder.classes_)
    n_classes = len(classes)

    drop_train = [c for c in ["id", "class"] if c in train.columns]
    drop_test = [c for c in ["id"] if c in test.columns]
    x_train, x_test, cat_cols = preprocess(train.drop(columns=drop_train), test.drop(columns=drop_test))

    print(f"Features: {x_train.shape[1]}  Categoricals: {cat_cols}  Classes: {classes}")

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 15,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    oof_proba = np.zeros((len(train), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(test), n_classes), dtype=np.float32)
    fold_rows = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), 1):
        model = LGBMClassifier(**params)
        model.fit(
            x_train.iloc[tr_idx], y[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(150, verbose=False), log_evaluation(250)],
        )

        va_p = model.predict_proba(x_train.iloc[va_idx])
        te_p = model.predict_proba(x_test)
        oof_proba[va_idx] = va_p
        test_proba += te_p / skf.get_n_splits()

        acc = accuracy_score(y[va_idx], va_p.argmax(axis=1))
        f1m = f1_score(y[va_idx], va_p.argmax(axis=1), average="macro")
        fold_rows.append({"fold": fold, "accuracy": acc, "f1_macro": f1m, "best_iteration": model.best_iteration_})
        print(f"fold={fold:2d} accuracy={acc:.6f} f1_macro={f1m:.6f} trees={model.best_iteration_}")

    oof_acc = accuracy_score(y, oof_proba.argmax(axis=1))
    oof_f1 = f1_score(y, oof_proba.argmax(axis=1), average="macro")
    print(f"\nOOF accuracy={oof_acc:.6f}  f1_macro={oof_f1:.6f}")

    np.save(MODELS / "lgbv5_oof_proba.npy", oof_proba)
    np.save(MODELS / "lgbv5_test_proba.npy", test_proba)

    metrics = {
        "model": "lgbv5",
        "oof_accuracy": oof_acc,
        "oof_f1_macro": oof_f1,
        "n_features": int(x_train.shape[1]),
        "categorical_features": cat_cols,
        "folds": fold_rows,
        "feature_blocks": [
            "redshift signed-log/fixed-bin/quantile-bin/quantile-transform",
            "redshift-bin interactions with colors and magnitude statistics",
            "full color indices plus adjacent 4-color ratios",
            "optional Petrosian/fiber morphology ratios plus safe morphology proxies",
            "RA/Dec sinusoidal harmonics, harmonic cross-terms, and galactic coordinates",
            "photometric SED residual/noise proxies",
            "STAR/QSO/GALAXY locus heuristic distances",
        ],
    }
    (MODELS / "lgbv5_metrics.json").write_text(json.dumps(metrics, indent=2))

    pred_labels = target_encoder.inverse_transform(test_proba.argmax(axis=1))
    sub = sample.copy()
    sub.iloc[:, 1] = pred_labels
    sub_path = SUBMISSIONS / f"lgbv5_oof{oof_acc:.5f}.csv"
    sub.to_csv(sub_path, index=False)
    print("Saved lgbv5 OOF/test probabilities, metrics, and submission CSV. Not submitted automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
