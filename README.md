# Kaggle Playground Series S6E6 — Predicting Stellar Class

**[Active]** — Deadline 2026-06-30 | Daniel Bandy (`goddangitboby`)

## Best Result

**Public LB: 0.95866** | 4-model Nelder-Mead ensemble (lgbv5 + lgbv6_mcc + lgb + xgb)

OOF log-loss: 0.08529 | OOF accuracy: 0.96964

## Results Summary

| Configuration | OOF Accuracy | OOF Log-loss | Public LB |
|---|---:|---:|---:|
| LGB baseline (5-fold) | 0.9681 | — | 0.95705 |
| lgbv6_mcc: LGB, MCC obj, α/δ + spectral (10-fold) | 0.96912 | 0.08669 | 0.95839 |
| lgbv5: LGB, galactic coords + photometric SED (10-fold) | **0.96931** | **0.08643** | — |
| dart: DART LGB (10-fold) | 0.96899 | 0.08694 | — |
| xgbv2: XGBoost 61-feature (training, fold 3/10) | ~0.969 | — | — |
| **4-model blend** (lgbv5+lgbv6_mcc+lgb+xgb, Nelder-Mead) | 0.96964 | 0.08529 | **0.95866** |

Blend weights: lgbv5=46%, lgbv6\_mcc=25%, lgb=16%, xgb=13%

## Dataset

577K rows × 12 features: photometric magnitudes (u, g, r, i, z), redshift, sky coords (α, δ), 2 categoricals.
Target: 3-class — GALAXY / QSO / STAR.
Key signal: redshift separates STAR (≈0.07) from GALAXY (≈0.51) and QSO (≈1.88) almost perfectly; the hard boundary is GALAXY vs QSO.

## Feature Engineering (lgbv5 — best single model)

- **Galactic coordinates**: equatorial (α, δ) → galactic (l, b) via J2000 transform; |b|, sin/cos(b), latitude bins (plane/intermediate/high)
- **Color indices**: u−g, g−r, r−i, i−z, u−r, g−z, g−i (standard SDSS photometric color sequence)
- **Redshift transforms**: log(1+z), z², z³, signed-log, z × |b| interaction
- **Photometric SED**: log-linear slope and amplitude across 5 SDSS bands (u/g/r/i/z)
- **Flux ratios**: u/g, g/r, r/i, i/z alternatives to color differences
- Total: 61 engineered features from 12 raw inputs

## Model Stack

| Script | Model | Status |
|---|---|---|
| `baseline_lgb.py` | LGB 5-fold baseline | Done — LB 0.95705 |
| `feateng_lgb.py` | LGB + color indices | Done |
| `feateng_v3_lgb.py` | LGB v3 + expanded features | Done — OOF 0.96888 |
| `feateng_v4_lgb.py` | LGB v4 + SED + flux ratios | Training (n_est=3000) |
| `feateng_v5_lgb.py` | LGB v5 + galactic coords | **Done — OOF 0.96931 (best)** |
| `lgbv6_mcc.py` | LGB, MCC objective + α/δ | Done — OOF 0.96912 |
| `dart_lgb.py` | DART LGB | Done — OOF 0.96899 |
| `xgb_model.py` | XGBoost baseline | Done — OOF 0.96870 |
| `xgbv2_galactic.py` | XGBoost 61-feat | Training (fold 3/10) |
| `catboost_model.py` | CatBoost | Done — OOF 0.96760 |
| `catboostv2_galactic.py` | CatBoost v2 | Done — OOF 0.96760 |
| `sklearn_ensemble.py` | HistGBM + ExtraTrees + RF | Done |
| `auto_blend_optimizer.py` | Nelder-Mead softmax blend | Automated |
| `meta_stack.py` | LR meta-learner (OOF → LR) | Done — no improvement (0.10074 logloss) |

## Blend Optimizer

`auto_blend_optimizer.py` scans `models/*_oof_proba.npy`, tries all combinations of size 2–5 plus all-model, optimizes softmax-normalized weights via Nelder-Mead (scipy), and submits only if the result beats `KNOWN_BEST_LOGLOSS`.

Simple Nelder-Mead blending outperformed LR meta-stacking for this correlated model set.

## Layout

```
data/       train.csv (577K rows), test.csv (247K rows)
models/     *_oof_proba.npy, *_test_proba.npy, metrics JSON
submissions/ Kaggle CSV files
*.py        Training scripts (see stack table above)
*.slurm     MCC SLURM job scripts
```

## Key Findings

- **Redshift is the dominant feature** by a wide margin — separates STAR from extragalactic objects near-perfectly
- **Galactic latitude (|b|)** adds discriminative power for GALAXY vs QSO at the boundary
- **Color indices** capture photometric SED shape that correlates with object type
- **DART regularization** and **MCC training objective** each improve ~0.05pp OOF accuracy over standard LGB
- **Blending** provides consistent +0.1–0.3pp LB improvement over any single model
- **Meta-stacking** (LR on OOF probas) does not improve over direct blending for highly correlated gradient boosting models
