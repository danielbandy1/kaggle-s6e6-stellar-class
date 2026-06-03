#!/usr/bin/env python3
"""Auto-discover and optimize S6E6 probability blends.

Scans models/*_oof_proba.npy, validates paired *_test_proba.npy files, tries
all blend combinations of size 2 through 5 plus an all-model blend, optimizes
softmax-normalized weights with scipy.optimize.minimize, and submits the best
log-loss blend to Kaggle.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path("/home/daniel/kaggle/s6e6")
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
COMP = "playground-series-s6e6"
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]
KNOWN_BEST_LOGLOSS = 0.08528656  # lgbv5+lgbv6_mcc+lgb+xgb, LB 0.95866 — only submit if better
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class ModelProbas:
    name: str
    oof_path: Path
    test_path: Path
    oof: np.ndarray
    test: np.ndarray


@dataclass
class BlendResult:
    combo_name: str
    names: tuple[str, ...]
    weights: np.ndarray
    logloss: float
    accuracy: float
    success: bool
    nit: int
    elapsed_sec: float


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def clean_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 1e-15, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return proba / row_sums


def load_labels() -> tuple[np.ndarray, pd.Series]:
    train = pd.read_csv(DATA / "train.csv", usecols=["class"])
    labels = train["class"].astype(str).map(CLASS_TO_ID)
    if labels.isna().any():
        unknown = sorted(set(train["class"].astype(str)) - set(CLASS_NAMES))
        raise ValueError(f"Unexpected class labels in train.csv: {unknown}")
    test_ids = pd.read_csv(DATA / "test.csv", usecols=["id"])["id"]
    return labels.to_numpy(dtype=np.int64), test_ids


def discover_models(expected_oof_shape: tuple[int, int], expected_test_rows: int) -> list[ModelProbas]:
    discovered: list[ModelProbas] = []
    print(f"Scanning {MODELS} for *_oof_proba.npy files", flush=True)

    EXCLUDE = {"best_blend", "blend_opt", "stack", "stacking", "pseudo"}
    for oof_path in sorted(MODELS.glob("*_oof_proba.npy")):
        name = oof_path.name[: -len("_oof_proba.npy")]
        if any(ex in name for ex in EXCLUDE):
            print(f"skip {name}: blend/stack artifact", flush=True)
            continue
        test_path = MODELS / f"{name}_test_proba.npy"

        try:
            oof = np.load(oof_path)
        except Exception as exc:
            print(f"WARN skip {name}: could not load OOF file {oof_path}: {exc}", flush=True)
            continue

        if oof.shape != expected_oof_shape:
            print(
                f"WARN skip {name}: OOF shape {oof.shape} != {expected_oof_shape}",
                flush=True,
            )
            continue

        if not test_path.exists():
            print(f"WARN skip {name}: missing paired test file {test_path}", flush=True)
            continue

        try:
            test = np.load(test_path)
        except Exception as exc:
            print(f"WARN skip {name}: could not load test file {test_path}: {exc}", flush=True)
            continue

        expected_test_shape = (expected_test_rows, expected_oof_shape[1])
        if test.shape != expected_test_shape:
            print(
                f"WARN skip {name}: test shape {test.shape} != {expected_test_shape}",
                flush=True,
            )
            continue

        oof = clean_proba(oof).astype(np.float32)
        test = clean_proba(test).astype(np.float32)
        discovered.append(ModelProbas(name=name, oof_path=oof_path, test_path=test_path, oof=oof, test=test))
        print(f"Loaded {name}: OOF {oof.shape}, test {test.shape}", flush=True)

    return discovered


def blend_arrays(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    blended = np.zeros_like(arrays[0], dtype=np.float64)
    for weight, arr in zip(weights, arrays, strict=True):
        blended += float(weight) * arr
    return clean_proba(blended)


def optimize_combo(models: list[ModelProbas], y: np.ndarray, combo_name: str) -> BlendResult:
    names = tuple(model.name for model in models)
    arrays = [model.oof for model in models]
    start = time.time()

    def objective(x: np.ndarray) -> float:
        weights = softmax(x)
        blended = blend_arrays(arrays, weights)
        return float(log_loss(y, blended, labels=np.arange(len(CLASS_NAMES))))

    init = np.zeros(len(models), dtype=np.float64)
    result = minimize(
        objective,
        init,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
    )

    weights = softmax(result.x)
    blended = blend_arrays(arrays, weights)
    blend_loss = float(log_loss(y, blended, labels=np.arange(len(CLASS_NAMES))))
    blend_acc = float(accuracy_score(y, blended.argmax(axis=1)))
    elapsed = time.time() - start

    print(
        f"{combo_name}: logloss={blend_loss:.8f} acc={blend_acc:.6f} "
        f"weights={format_weights(names, weights)} success={result.success} "
        f"nit={getattr(result, 'nit', -1)} elapsed={elapsed:.1f}s",
        flush=True,
    )

    return BlendResult(
        combo_name=combo_name,
        names=names,
        weights=weights,
        logloss=blend_loss,
        accuracy=blend_acc,
        success=bool(result.success),
        nit=int(getattr(result, "nit", -1)),
        elapsed_sec=float(elapsed),
    )


def format_weights(names: tuple[str, ...], weights: np.ndarray) -> str:
    return ", ".join(f"{name}={weight:.5f}" for name, weight in zip(names, weights, strict=True))


def combo_plan(models: list[ModelProbas]) -> list[tuple[str, tuple[int, ...]]]:
    n = len(models)
    combos: list[tuple[str, tuple[int, ...]]] = []
    for size in range(2, min(5, n) + 1):
        for indices in itertools.combinations(range(n), size):
            names = "+".join(models[i].name for i in indices)
            combos.append((names, indices))

    all_indices = tuple(range(n))
    combos.append(("ALL_MODELS", all_indices))
    return combos


def save_best_blend(best: BlendResult, models_by_name: dict[str, ModelProbas], test_ids: pd.Series) -> Path:
    selected = [models_by_name[name] for name in best.names]
    oof_blend = blend_arrays([model.oof for model in selected], best.weights).astype(np.float32)
    test_blend = blend_arrays([model.test for model in selected], best.weights).astype(np.float32)

    np.save(MODELS / "best_blend_oof_proba.npy", oof_blend)
    np.save(MODELS / "best_blend_test_proba.npy", test_blend)

    pred_classes = [CLASS_NAMES[idx] for idx in test_blend.argmax(axis=1)]
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS / f"best_blend_{best.logloss:.5f}.csv"
    pd.DataFrame({"id": test_ids, "class": pred_classes}).to_csv(sub_path, index=False)
    print(f"Saved best blend OOF/test arrays to {MODELS}", flush=True)
    print(f"Saved submission: {sub_path}", flush=True)
    return sub_path


def submit_best(best: BlendResult, sub_path: Path) -> None:
    model_names = "+".join(best.names)
    weight_text = "[" + ",".join(f"{w:.5f}" for w in best.weights) + "]"
    msg = f"Auto-blend optimizer: {model_names} weights={weight_text}"
    if len(msg) > 240:
        msg = f"Auto-blend optimizer: {best.combo_name} logloss={best.logloss:.5f}"
    print(f"Submitting to Kaggle: {msg}", flush=True)
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMP, "-f", str(sub_path), "-m", msg],
        check=False,
    )


def print_top_results(results: list[BlendResult], limit: int = 15) -> None:
    print("\nTop blend results by OOF log-loss:", flush=True)
    print("rank logloss accuracy combo weights success nit elapsed_sec", flush=True)
    for rank, result in enumerate(sorted(results, key=lambda r: r.logloss)[:limit], start=1):
        print(
            f"{rank:02d} {result.logloss:.8f} {result.accuracy:.6f} "
            f"{result.combo_name} {format_weights(result.names, result.weights)} "
            f"{result.success} {result.nit} {result.elapsed_sec:.1f}",
            flush=True,
        )


def write_results_json(results: list[BlendResult], best: BlendResult) -> None:
    payload = {
        "best": {
            "combo_name": best.combo_name,
            "models": list(best.names),
            "weights": [float(w) for w in best.weights],
            "logloss": best.logloss,
            "accuracy": best.accuracy,
        },
        "results": [
            {
                "combo_name": result.combo_name,
                "models": list(result.names),
                "weights": [float(w) for w in result.weights],
                "logloss": result.logloss,
                "accuracy": result.accuracy,
                "success": result.success,
                "nit": result.nit,
                "elapsed_sec": result.elapsed_sec,
            }
            for result in sorted(results, key=lambda r: r.logloss)
        ],
    }
    out_path = MODELS / "auto_blend_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved optimizer results: {out_path}", flush=True)


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    y, test_ids = load_labels()
    expected_oof_shape = (len(y), len(CLASS_NAMES))
    print(f"Loaded labels: {expected_oof_shape[0]} rows, class order {CLASS_NAMES}", flush=True)

    models = discover_models(expected_oof_shape, expected_test_rows=len(test_ids))
    if len(models) < 2:
        print(f"Need at least two complete model probability pairs; found {len(models)}.", flush=True)
        return 1

    print(f"Available models for blending: {[model.name for model in models]}", flush=True)
    combos = combo_plan(models)
    print(f"Trying {len(combos)} blend combos: sizes 2-5 plus ALL_MODELS", flush=True)

    results: list[BlendResult] = []
    for combo_name, indices in combos:
        selected = [models[idx] for idx in indices]
        results.append(optimize_combo(selected, y, combo_name))

    results.sort(key=lambda result: result.logloss)
    print_top_results(results, limit=15)

    best = results[0]
    print(
        f"\nWinner: {best.combo_name} logloss={best.logloss:.8f} "
        f"accuracy={best.accuracy:.6f} weights={format_weights(best.names, best.weights)}",
        flush=True,
    )

    models_by_name = {model.name: model for model in models}
    sub_path = save_best_blend(best, models_by_name, test_ids)
    write_results_json(results, best)
    if best.logloss < KNOWN_BEST_LOGLOSS:
        submit_best(best, sub_path)
    else:
        print(
            f"No submission: logloss {best.logloss:.8f} >= known best {KNOWN_BEST_LOGLOSS:.8f}",
            flush=True,
        )

    print(
        f"\nSummary: tried {len(results)} combos from {len(models)} complete models. "
        f"Best={best.combo_name} logloss={best.logloss:.8f}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
