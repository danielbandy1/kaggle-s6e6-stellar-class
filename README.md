# Kaggle Playground Series S6E6 — Predicting Stellar Class

**[Active]** — Deadline 2026-06-30 | Daniel Bandy (`goddangitboby`)

## Results

| Model | OOF Accuracy | OOF F1 (macro) | Public LB | Notes |
|---|---:|---:|---:|---|
| LGB baseline (5-fold) | 0.9681 | 0.9571 | 0.95705 | n_estimators=1200, lr=0.03 |
| LGB + color indices (10-fold) | — | — | *in progress on RED* | u-g, g-r, r-i, i-z, log-redshift |
| LGB + Optuna best params (10-fold) | — | — | *pending MCC job 35224179* | 100-trial Optuna search |
| LGB + XGB blend | — | — | *pending* | Grid-search optimal weight |

Top LB: 0.97049 (gap: ~0.013). Target: ≥0.970.

## Dataset

577K rows × 12 features: photometric magnitudes (u, g, r, i, z), redshift, sky coords (α, δ), 2 categoricals.
Target: 3-class stellar classification — GALAXY / QSO / STAR.
Key signal: redshift (STAR≈0.07, GALAXY≈0.51, QSO≈1.88).

## Feature Engineering

Added astrophysics color indices (u−g, g−r, r−i, i−z, u−r, g−z), log-redshift, redshift², magnitude statistics (mean, std, range), and sky coordinate trig features. See `feateng_lgb.py`.

## Stack

| Script | Purpose |
|---|---|
| `baseline_lgb.py` | 5-fold LGB, submitted LB 0.95705 |
| `feateng_lgb.py` | 10-fold LGB + color indices |
| `optuna_lgb_tune.py` | 100-trial Optuna search (MCC) |
| `retrain_best.py` | Retrain with Optuna params + color indices; saves OOF/test probas |
| `xgb_model.py` | XGBoost companion, saves probas for blending |
| `blend_ensemble.py` | LGB+XGB grid-search blend, auto-submits |
| `mcc_tune.slurm` | SLURM job (coa_ich248_uksr, short partition, 3.5hr) |
| `collect-mcc-results.sh` | Sync MCC results → retrain → submit pipeline |

## Layout

- `data/` — train.csv (96MB), test.csv (40MB), sample_submission.csv
- `models/` — OOF predictions, metrics JSON, tuned params, proba arrays
- `submissions/` — Kaggle submission CSVs
- `eda_summary.txt` — data inspection summary

## Reproducing the Baseline

```bash
cd /home/daniel/kaggle/s6e6
python3 baseline_lgb.py
```

## Running the Full Pipeline (after MCC job completes)

```bash
# 1. Collect Optuna results and retrain
bash ~/bin/collect-mcc-results.sh s6e6

# 2. Run XGBoost companion
python3 xgb_model.py

# 3. Blend and submit
python3 blend_ensemble.py
```
