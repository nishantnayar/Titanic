"""
Pre-compute counterfactual results for the 5 archetype passengers across all
4 regulatory frameworks × 6 CF methods.

Run once (or after retraining the model):
    python -m src.precompute_stories_cfs

Output: data/stories_cf_cache.pkl
  Structure: Dict[passenger_id, Dict[fw_key, Dict[method, List[Counterfactual]]]]
"""

import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.features import engineer_features
from src.stories import ARCHETYPES
from src.counterfactual_methods import CounterfactualEngine
from src.regulatory_framework import FRAMEWORKS, get_framework

CACHE_PATH = Path("data/stories_cf_cache.pkl")


def main():
    print("Loading artifacts...")
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    stats = joblib.load("models/train_stats.pkl")
    X_train = joblib.load("models/X_train.pkl")
    train_df = pd.read_csv("data/train.csv")
    engine = CounterfactualEngine(model, X_train, scaler)

    # Pre-instantiate all framework objects (loads IsolationForest once each)
    fw_objects = {k: get_framework(k, X_train) for k in FRAMEWORKS}

    cache = {}

    for archetype in ARCHETYPES:
        pid = archetype["id"]
        name = archetype["name"]
        print(f"\n{'='*60}")
        print(f"Passenger: {name} (id={pid})")

        row = train_df[train_df["PassengerId"] == pid].iloc[0]
        df_single = pd.DataFrame([row])
        X_raw = engineer_features(
            df_single,
            train_age_mean=stats["age_mean"],
            train_age_std=stats["age_std"],
        )
        X_inst = scaler.transform(X_raw)[0]

        cache[pid] = {}

        for fw_key, fw in fw_objects.items():
            print(f"  Framework: {fw.name}")
            cache[pid][fw_key] = {}

            for method in CounterfactualEngine.METHOD_NAMES:
                label = CounterfactualEngine.METHOD_LABELS[method]
                try:
                    cfs = engine.generate(X_inst, method=method, framework=fw, n=1)
                    cache[pid][fw_key][method] = cfs
                    status = f"{len(cfs)} CF(s)" if cfs else "0 CFs"
                except Exception as exc:
                    cache[pid][fw_key][method] = []
                    status = f"ERROR: {exc}"
                print(f"    {label}: {status}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, CACHE_PATH)
    print(f"\nSaved cache to {CACHE_PATH}")
    print(
        f"Covers: {len(ARCHETYPES)} passengers × "
        f"{len(FRAMEWORKS)} frameworks × "
        f"{len(CounterfactualEngine.METHOD_NAMES)} methods"
    )


if __name__ == "__main__":
    main()
