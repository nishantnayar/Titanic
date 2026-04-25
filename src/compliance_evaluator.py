"""
Automated compliance evaluator for counterfactual explanation methods.

Runs all 6 methods x 4 regulatory frameworks against a held-out test
slice of the training data and computes 6 evaluation metrics:

    Validity     – % of CFs that flip the prediction
    Proximity    – mean L1 distance to original
    Sparsity     – mean number of features changed
    Plausibility – mean IsolationForest score
    Actionability – % of CFs respecting immutability constraints
    Diversity    – mean pairwise L1 distance across the k=3 CFs

Results are saved to data/regulatory_audit_results.csv.

Usage
-----
    python -m src.compliance_evaluator          # full audit (~2-5 min)
    python -m src.compliance_evaluator --quick  # 20-row sample
"""

from __future__ import annotations

import argparse
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import engineer_features
from src.regulatory_framework import (
    FRAMEWORKS,
    FRAMEWORK_DISPLAY_NAMES,
    get_framework,
)
from src.counterfactual_methods import CounterfactualEngine, Counterfactual

warnings.filterwarnings("ignore")

AUDIT_CSV = Path("data/regulatory_audit_results.csv")
N_CFS = 3


# ---------------------------------------------------------------------------
# Six evaluation metrics
# ---------------------------------------------------------------------------
def metric_validity(cfs: List[Counterfactual], original_pred: int) -> float:
    """Fraction of CFs that successfully flip the prediction."""
    if not cfs:
        return 0.0
    flipped = sum(1 for cf in cfs if cf.prediction != original_pred)
    return flipped / len(cfs)


def metric_proximity(cfs: List[Counterfactual]) -> float:
    """Mean L1 distance from original instance."""
    if not cfs:
        return np.nan
    return float(np.mean([cf.distance for cf in cfs]))


def metric_sparsity(cfs: List[Counterfactual]) -> float:
    """Mean number of features changed per CF."""
    if not cfs:
        return np.nan
    return float(np.mean([cf.n_changes for cf in cfs]))


def metric_plausibility(cfs: List[Counterfactual]) -> float:
    """Mean IsolationForest plausibility score (0-1, higher is better)."""
    if not cfs:
        return np.nan
    return float(np.mean([cf.plausibility_score for cf in cfs]))


def metric_actionability(cfs: List[Counterfactual]) -> float:
    """
    Fraction of CFs that respect immutability constraints.
    is_compliant only fails for immutability or plausibility violations —
    here we focus purely on immutability (no violations at all).
    """
    if not cfs:
        return 0.0
    immut_ok = sum(
        1 for cf in cfs
        if not any("Immutable" in v for v in cf.violations)
    )
    return immut_ok / len(cfs)


def metric_diversity(cfs: List[Counterfactual]) -> float:
    """
    Mean pairwise L1 distance across the k CFs.
    Higher = more diverse set of recourse options.
    """
    if len(cfs) < 2:
        return 0.0
    pairs = list(combinations(cfs, 2))
    dists = [
        float(np.sum(np.abs(a.features - b.features)))
        for a, b in pairs
    ]
    return float(np.mean(dists))


def evaluate_single(
    cfs: List[Counterfactual],
    original_pred: int,
) -> Dict[str, float]:
    """Compute all 6 metrics for one (instance, method, framework) combo."""
    return {
        "validity": metric_validity(cfs, original_pred),
        "proximity": metric_proximity(cfs),
        "sparsity": metric_sparsity(cfs),
        "plausibility": metric_plausibility(cfs),
        "actionability": metric_actionability(cfs),
        "diversity": metric_diversity(cfs),
    }


# ---------------------------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------------------------
def evaluate_method(
    engine: CounterfactualEngine,
    method: str,
    framework_key: str,
    X_test: np.ndarray,
    X_train: np.ndarray,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run one method × one framework over the full test set.
    Returns averaged metrics across all test instances.
    """
    fw = get_framework(framework_key, X_train)
    all_metrics: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    for i, instance in enumerate(X_test):
        original_pred = int(engine.model.predict(instance.reshape(1, -1))[0])
        try:
            cfs = engine.generate(instance, method=method, framework=fw, n=N_CFS)
        except Exception:
            cfs = []
        all_metrics.append(evaluate_single(cfs, original_pred))

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            pct = (i + 1) / len(X_test)
            print(f"  {method}/{framework_key}: {i+1} rows ({elapsed:.1f}s, {pct:.0%})")

    if not all_metrics:
        return {}

    df = pd.DataFrame(all_metrics)
    return {col: float(df[col].mean()) for col in df.columns}


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------
def run_full_audit(
    n_test_rows: int = 150,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all 6 methods x 4 frameworks on `n_test_rows` test instances.
    Saves results to data/regulatory_audit_results.csv.

    Returns the results DataFrame.
    """
    # Load artifacts
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    stats = joblib.load("models/train_stats.pkl")
    X_train = joblib.load("models/X_train.pkl")

    # Build test slice from training CSV
    train_df = pd.read_csv("data/train.csv")
    _, test_df = train_test_split(train_df, test_size=0.25, random_state=42)

    X_test_raw = engineer_features(
        test_df,
        train_age_mean=stats["age_mean"],
        train_age_std=stats["age_std"],
    )
    X_test = scaler.transform(X_test_raw)

    if n_test_rows and n_test_rows < len(X_test):
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_test), n_test_rows, replace=False)
        X_test = X_test[idx]

    # Fit IsolationForest once (shared)
    from src.regulatory_framework import _load_or_fit_isolation_forest
    _load_or_fit_isolation_forest(X_train)

    engine = CounterfactualEngine(model, X_train, scaler)

    methods = CounterfactualEngine.METHOD_NAMES
    frameworks = list(FRAMEWORKS.keys())

    rows: List[dict] = []
    total = len(methods) * len(frameworks)
    done = 0

    for method in methods:
        for fw_key in frameworks:
            done += 1
            if verbose:
                label = CounterfactualEngine.METHOD_LABELS[method]
                fw_label = FRAMEWORK_DISPLAY_NAMES[fw_key]
                print(
                    f"[{done}/{total}] {label}  x  {fw_label} "
                    f"({n_test_rows} rows)..."
                )
            metrics = evaluate_method(
                engine, method, fw_key, X_test, X_train, verbose=verbose
            )
            rows.append(
                {
                    "method": method,
                    "method_label": CounterfactualEngine.METHOD_LABELS[method],
                    "framework": fw_key,
                    "framework_label": FRAMEWORK_DISPLAY_NAMES[fw_key],
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(AUDIT_CSV, index=False)
    if verbose:
        print(f"\nAudit complete. Saved to {AUDIT_CSV}")
    return results


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Titanic CF compliance audit."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 20-row sample instead of full test set.",
    )
    args = parser.parse_args()
    n_rows = 20 if args.quick else 150
    run_full_audit(n_test_rows=n_rows, verbose=True)
