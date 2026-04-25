"""
Unified API for 6 counterfactual explanation methods.

Methods
-------
1. nearest_neighbor  - Nearest Unlike Neighbour (custom)
2. nice              - NICE with sparsity/proximity/plausibility/none
3. dice_genetic      - DiCE genetic algorithm backend
4. dice_kdtree       - DiCE KDTree backend
5. ocean             - OCEAN CP solver (fallback on Windows)
6. feature_tweaking  - RF decision-path traversal (custom)

All methods share a single interface::

    engine = CounterfactualEngine(model, X_train, scaler)
    cfs = engine.generate(instance, method='nice', framework=fw, n=3)
    # Returns List[Counterfactual]
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.regulatory_framework import (
    RegulatoryFramework,
    ComplianceResult,
    UnconstrainedFramework,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Counterfactual dataclass
# ---------------------------------------------------------------------------
@dataclass
class Counterfactual:
    features: np.ndarray        # scaled feature vector
    prediction: int             # model's predicted class
    distance: float             # L1 distance from original
    n_changes: int              # number of features changed
    plausibility_score: float   # IsolationForest score (0-1)
    is_compliant: bool          # passes framework constraints
    violations: List[str]       # violation descriptions
    method: str = ""            # generating method name
    elapsed_ms: float = 0.0     # wall-clock time in ms


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _count_changes(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> int:
    return int(np.sum(~np.isclose(a, b, atol=atol)))


def _validate_cf(
    cf_vec: np.ndarray,
    original: np.ndarray,
    model,
    framework: RegulatoryFramework,
    method: str,
    elapsed_ms: float,
) -> Counterfactual:
    """Wrap a raw CF vector into an annotated Counterfactual dataclass."""
    pred = int(model.predict(cf_vec.reshape(1, -1))[0])
    result: ComplianceResult = framework.validate(cf_vec, original)
    return Counterfactual(
        features=cf_vec,
        prediction=pred,
        distance=_l1_distance(original, cf_vec),
        n_changes=_count_changes(original, cf_vec),
        plausibility_score=result.plausibility_score,
        is_compliant=result.is_compliant,
        violations=result.violations,
        method=method,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Feature Tweaking helpers
# ---------------------------------------------------------------------------
def _get_leaf_paths(tree_) -> dict:
    """DFS over sklearn tree_ → {leaf_id: [(feat, dir, threshold)]}."""
    paths: dict = {}

    def dfs(node: int, path: list) -> None:
        if tree_.children_left[node] == -1:
            paths[node] = path
            return
        feat = int(tree_.feature[node])
        thresh = float(tree_.threshold[node])
        dfs(tree_.children_left[node], path + [(feat, "left", thresh)])
        dfs(tree_.children_right[node], path + [(feat, "right", thresh)])

    dfs(0, [])
    return paths


def _feature_tweak_single_tree(
    estimator,
    instance: np.ndarray,
    target_class: int,
    immutable_idx: List[int],
) -> tuple:
    """Return (best_cf, best_cost) for one decision tree."""
    tree_ = estimator.tree_
    leaf_paths = _get_leaf_paths(tree_)

    if tree_.value.ndim == 3:
        leaf_classes = np.argmax(tree_.value[:, 0, :], axis=1)
    else:
        leaf_classes = np.argmax(tree_.value, axis=1)

    best_cf: Optional[np.ndarray] = None
    best_cost = np.inf

    for leaf_id, path in leaf_paths.items():
        if leaf_classes[leaf_id] != target_class:
            continue

        cf = instance.copy()
        cost = 0.0
        feasible = True

        for feat, direction, thresh in path:
            if direction == "left":  # need cf[feat] <= thresh
                if cf[feat] > thresh:
                    if feat in immutable_idx:
                        feasible = False
                        break
                    new_val = thresh - 1e-4
                    cost += abs(new_val - instance[feat])
                    cf[feat] = new_val
            else:  # direction == "right", need cf[feat] > thresh
                if cf[feat] <= thresh:
                    if feat in immutable_idx:
                        feasible = False
                        break
                    new_val = thresh + 1e-4
                    cost += abs(new_val - instance[feat])
                    cf[feat] = new_val

        if feasible and cost < best_cost:
            best_cost = cost
            best_cf = cf

    return best_cf, best_cost


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class CounterfactualEngine:
    """Unified interface to all 6 counterfactual methods."""

    METHOD_NAMES = [
        "nearest_neighbor",
        "nice",
        "dice_genetic",
        "dice_kdtree",
        "ocean",
        "feature_tweaking",
    ]

    METHOD_LABELS = {
        "nearest_neighbor": "Nearest Unlike Neighbour",
        "nice": "NICE",
        "dice_genetic": "DiCE-Genetic",
        "dice_kdtree": "DiCE-KDTree",
        "ocean": "OCEAN",
        "feature_tweaking": "Feature Tweaking",
    }

    def __init__(self, model, X_train: np.ndarray, scaler=None) -> None:
        self.model = model
        self.X_train = X_train
        self.scaler = scaler
        self._train_df: Optional[pd.DataFrame] = None
        self._dice_data = None
        self._dice_model = None

    # ------------------------------------------------------------------
    def generate(
        self,
        instance: np.ndarray,
        method: str = "nearest_neighbor",
        framework: Optional[RegulatoryFramework] = None,
        n: int = 3,
    ) -> List[Counterfactual]:
        """Generate up to `n` counterfactuals using the named method."""
        if framework is None:
            framework = UnconstrainedFramework()

        dispatch = {
            "nearest_neighbor": self._nearest_neighbor,
            "nice": self._nice,
            "dice_genetic": self._dice_genetic,
            "dice_kdtree": self._dice_kdtree,
            "ocean": self._ocean,
            "feature_tweaking": self._feature_tweaking,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {self.METHOD_NAMES}"
            )

        t0 = time.perf_counter()
        raw_cfs = fn(instance, framework, n)
        elapsed = (time.perf_counter() - t0) * 1000.0

        results: List[Counterfactual] = []
        for cf_vec in raw_cfs:
            if cf_vec is None:
                continue
            results.append(
                _validate_cf(
                    np.array(cf_vec, dtype=float),
                    instance,
                    self.model,
                    framework,
                    method,
                    elapsed,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Method 1 — Nearest Unlike Neighbour
    # ------------------------------------------------------------------
    def _nearest_neighbor(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        from sklearn.metrics.pairwise import euclidean_distances

        instance_pred = self.model.predict(instance.reshape(1, -1))[0]
        train_preds = self.model.predict(self.X_train)
        opposite_X = self.X_train[train_preds != instance_pred]

        if len(opposite_X) == 0:
            return []

        immut_idx = framework.immutable_indices()
        dists = euclidean_distances(instance.reshape(1, -1), opposite_X)[0]

        # Large penalty for immutability violations
        for i, cand in enumerate(opposite_X):
            for idx in immut_idx:
                if not np.isclose(cand[idx], instance[idx], atol=1e-6):
                    dists[i] += 1e6

        ranked = np.argsort(dists)
        selected: List[int] = []
        for idx in ranked:
            cand = opposite_X[idx]
            if all(
                np.linalg.norm(cand - opposite_X[s]) > 0.05 for s in selected
            ):
                selected.append(idx)
            if len(selected) == n:
                break

        return [opposite_X[i] for i in selected]

    # ------------------------------------------------------------------
    # Method 2 — NICE
    # ------------------------------------------------------------------
    def _nice(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        try:
            from nice import NICE
        except ImportError:
            return self._nearest_neighbor(instance, framework, n)

        y_train = self.model.predict(self.X_train)
        results: List[np.ndarray] = []
        seen: List[np.ndarray] = []

        for opt in ["sparsity", "proximity", "plausibility", "none"]:
            if len(results) >= n:
                break
            try:
                explainer = NICE(
                    predict_fn=lambda X: self.model.predict(X),
                    X_train=self.X_train,
                    cat_feat=[],
                    y_train=y_train,
                    optimization=opt,
                    justified_cf=True,
                )
                cf = np.array(
                    explainer.explain(instance.reshape(1, -1))
                ).flatten()
                if not any(np.allclose(cf, s, atol=1e-6) for s in seen):
                    seen.append(cf)
                    results.append(cf)
            except Exception:
                continue

        if len(results) < n:
            extra = self._nearest_neighbor(
                instance, framework, n - len(results)
            )
            results.extend(extra)

        return results[:n]

    # ------------------------------------------------------------------
    # Method 3 & 4 — DiCE (genetic / kdtree backends)
    # ------------------------------------------------------------------
    def _setup_dice(self) -> None:
        """Lazy initialisation of DiCE Data and Model objects."""
        if self._train_df is not None:
            return

        import dice_ml

        df = pd.DataFrame(self.X_train, columns=FEATURE_COLUMNS)
        df["Survived"] = self.model.predict(self.X_train).astype(int)
        self._train_df = df

        self._dice_data = dice_ml.Data(
            dataframe=df,
            continuous_features=FEATURE_COLUMNS,
            outcome_name="Survived",
        )
        self._dice_model = dice_ml.Model(model=self.model, backend="sklearn")

    def _dice_generate(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
        method: str,
    ) -> List[np.ndarray]:
        try:
            import dice_ml  # noqa: F401
        except ImportError:
            return self._nearest_neighbor(instance, framework, n)

        try:
            self._setup_dice()
            exp = dice_ml.Dice(
                self._dice_data, self._dice_model, method=method
            )

            current_pred = int(
                self.model.predict(instance.reshape(1, -1))[0]
            )
            target_pred = 1 - current_pred
            immut_names = list(
                framework.immutable_features.intersection(set(FEATURE_COLUMNS))
            )
            vary = [f for f in FEATURE_COLUMNS if f not in immut_names]

            query = pd.DataFrame([instance], columns=FEATURE_COLUMNS)
            dice_result = exp.generate_counterfactuals(
                query,
                total_CFs=n,
                desired_class=target_pred,
                features_to_vary=vary,
                verbose=False,
            )
            cfs_df = dice_result.cf_examples_list[0].final_cfs_df
            if cfs_df is None or len(cfs_df) == 0:
                return self._nearest_neighbor(instance, framework, n)

            return [
                row[FEATURE_COLUMNS].values.astype(float)
                for _, row in cfs_df.iterrows()
            ]
        except Exception:
            return self._nearest_neighbor(instance, framework, n)

    def _dice_genetic(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        return self._dice_generate(instance, framework, n, method="genetic")

    def _dice_kdtree(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        return self._dice_generate(instance, framework, n, method="kdtree")

    # ------------------------------------------------------------------
    # Method 5 — OCEAN (CP solver; falls back to plausibility-weighted NN)
    # ------------------------------------------------------------------
    def _ocean(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        try:
            from ocean.cp import ClassifierCounterfactualMilp  # noqa: F401
            return self._ocean_native(instance, framework, n)
        except Exception:
            return self._ocean_fallback(instance, framework, n)

    def _ocean_native(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        from ocean.cp import ClassifierCounterfactualMilp
        from ocean.feature import FeatureType

        feature_types = [FeatureType.Numeric] * len(FEATURE_COLUMNS)
        solver = ClassifierCounterfactualMilp(
            self.model,
            instance,
            feature_types=feature_types,
            immutable_features=framework.immutable_indices(),
        )
        cf = solver.solve()
        return [np.array(cf)] if cf is not None else []

    def _ocean_fallback(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        """Plausibility-weighted nearest-neighbour (OCEAN proxy)."""
        import joblib
        from pathlib import Path
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics.pairwise import euclidean_distances

        iso_path = Path("models/isolation_forest.pkl")
        iso = (
            joblib.load(iso_path)
            if iso_path.exists()
            else IsolationForest(contamination=0.1, random_state=42).fit(
                self.X_train
            )
        )

        instance_pred = self.model.predict(instance.reshape(1, -1))[0]
        train_preds = self.model.predict(self.X_train)
        opposite_X = self.X_train[train_preds != instance_pred]

        if len(opposite_X) == 0:
            return []

        dists = euclidean_distances(instance.reshape(1, -1), opposite_X)[0]
        plaus = np.clip(
            iso.decision_function(opposite_X) + 0.5, 0.0, 1.0
        )
        scores = dists * (2.0 - plaus)
        immut_idx = framework.immutable_indices()
        for i, cand in enumerate(opposite_X):
            for idx in immut_idx:
                if not np.isclose(cand[idx], instance[idx], atol=1e-6):
                    scores[i] += 1e6

        ranked = np.argsort(scores)
        selected: List[int] = []
        for idx in ranked:
            cand = opposite_X[idx]
            if all(
                np.linalg.norm(cand - opposite_X[s]) > 0.05 for s in selected
            ):
                selected.append(idx)
            if len(selected) == n:
                break

        return [opposite_X[i] for i in selected]

    # ------------------------------------------------------------------
    # Method 6 — Feature Tweaking (RF decision-path traversal)
    # ------------------------------------------------------------------
    def _feature_tweaking(
        self,
        instance: np.ndarray,
        framework: RegulatoryFramework,
        n: int,
    ) -> List[np.ndarray]:
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(self.model, RandomForestClassifier):
            return self._nearest_neighbor(instance, framework, n)

        current_pred = int(self.model.predict(instance.reshape(1, -1))[0])
        target_class = 1 - current_pred
        immut_idx = framework.immutable_indices()

        candidates: List[tuple] = []
        for estimator in self.model.estimators_:
            cf, cost = _feature_tweak_single_tree(
                estimator, instance, target_class, immut_idx
            )
            if cf is not None:
                pred = int(self.model.predict(cf.reshape(1, -1))[0])
                if pred == target_class:
                    candidates.append((cost, cf))

        candidates.sort(key=lambda x: x[0])
        results: List[np.ndarray] = []
        for _, cf in candidates:
            if len(results) >= n:
                break
            if not any(np.allclose(cf, r, atol=1e-4) for r in results):
                results.append(cf)

        if len(results) < n:
            extra = self._nearest_neighbor(
                instance, framework, n - len(results)
            )
            results.extend(extra)

        return results[:n]
