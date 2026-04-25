"""
Regulatory compliance framework classes for counterfactual evaluation.

Implements GDPR Article 22, CFPB Circular 2022-03, EU AI Act (High-Risk),
and an unconstrained research baseline.  Each framework defines:
  - Immutable features (cannot be changed in counterfactuals)
  - Plausibility requirements (IsolationForest-based)
  - Diversity / actionability requirements
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_IF_PATH = Path("models/isolation_forest.pkl")

# ---------------------------------------------------------------------------
# Immutable feature sets (in engineered feature space)
# ---------------------------------------------------------------------------
# GDPR Art 22: biological / personal identity attributes
GDPR_IMMUTABLE: set = {"Sex", "Age", "SibSp", "Parch"}
# CFPB 2022-03: protected classes in credit / financial decisions
CFPB_IMMUTABLE: set = {"Sex"}
# EU AI Act High-Risk: protected attributes + Title (social class proxy)
EU_AI_IMMUTABLE: set = {"Sex", "Title"}

# Column name → position index in FEATURE_COLUMNS
_FEAT_IDX: Dict[str, int] = {col: i for i, col in enumerate(FEATURE_COLUMNS)}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ComplianceResult:
    is_compliant: bool
    violations: List[str]
    plausibility_score: float


# ---------------------------------------------------------------------------
# IsolationForest helper
# ---------------------------------------------------------------------------
def _load_or_fit_isolation_forest(X_train: Optional[np.ndarray] = None):
    """Load cached IsolationForest or fit a new one on X_train."""
    from sklearn.ensemble import IsolationForest

    if _IF_PATH.exists():
        return joblib.load(_IF_PATH)

    if X_train is None:
        X_train = joblib.load("models/X_train.pkl")

    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X_train)
    _IF_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, _IF_PATH)
    return iso


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class RegulatoryFramework:
    """
    Abstract base for regulatory compliance frameworks.

    Subclasses set class-level defaults; instances can be re-used across
    many counterfactuals once constructed.
    """

    name: str = "Base"
    description: str = ""
    immutable_features: set = set()
    requires_plausibility: bool = False
    plausibility_threshold: float = 0.0
    requires_diversity: bool = False
    requires_actionable_reasons: bool = False

    def __init__(self, X_train: Optional[np.ndarray] = None) -> None:
        self._iso = None
        if self.requires_plausibility:
            self._iso = _load_or_fit_isolation_forest(X_train)

    # ------------------------------------------------------------------
    def plausibility_score(self, cf: np.ndarray) -> float:
        """
        Returns a 0–1 plausibility score for a counterfactual vector.
        Uses IsolationForest decision function, normalised to [0, 1].
        Higher = more plausible (more similar to training distribution).
        """
        if self._iso is None:
            return 1.0
        raw = self._iso.decision_function(cf.reshape(1, -1))[0]
        # decision_function ≈ [-0.5, +0.5]; shift to [0, 1]
        return float(np.clip(raw + 0.5, 0.0, 1.0))

    # ------------------------------------------------------------------
    def validate(
        self, cf: np.ndarray, original: np.ndarray
    ) -> ComplianceResult:
        """
        Check whether counterfactual `cf` complies with this framework's
        constraints given the `original` instance.
        """
        violations: List[str] = []

        for feat in self.immutable_features:
            idx = _FEAT_IDX.get(feat)
            if idx is not None and not np.isclose(
                cf[idx], original[idx], atol=1e-6
            ):
                violations.append(f"Immutable feature changed: {feat}")

        ps = self.plausibility_score(cf)
        if self.requires_plausibility and ps < self.plausibility_threshold:
            violations.append(
                f"Implausible CF (score={ps:.3f} "
                f"< threshold={self.plausibility_threshold})"
            )

        return ComplianceResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            plausibility_score=ps,
        )

    # ------------------------------------------------------------------
    def immutable_indices(self) -> List[int]:
        """Return column indices of immutable features."""
        return [
            _FEAT_IDX[f] for f in self.immutable_features if f in _FEAT_IDX
        ]


# ---------------------------------------------------------------------------
# Concrete frameworks
# ---------------------------------------------------------------------------
class GDPRFramework(RegulatoryFramework):
    """
    GDPR Article 22 — right to meaningful explanation for automated decisions.

    Constraints:
      - Sex, Age, SibSp, Parch are immutable (personal/biological attributes).
      - Counterfactuals must be plausible (IsolationForest score ≥ 0.3).
      - Diversity required (at least 3 distinct CFs).
    """

    name = "GDPR Article 22"
    description = (
        "GDPR Art. 22 requires that automated decisions affecting individuals "
        "come with meaningful explanations and actionable recourse that "
        "respects immutable personal attributes."
    )
    immutable_features = GDPR_IMMUTABLE
    requires_plausibility = True
    plausibility_threshold = 0.3
    requires_diversity = True
    requires_actionable_reasons = True


class CFPBFramework(RegulatoryFramework):
    """
    CFPB Circular 2022-03 — adverse action notices with specific reasons.

    Constraints:
      - Sex is immutable (protected class under ECOA).
      - Reasons must be specific and actionable.
      - No plausibility requirement (CFPB focuses on actionability).
    """

    name = "CFPB Circular 2022-03"
    description = (
        "CFPB Circular 2022-03 requires that credit decisions include "
        "specific, actionable adverse action reasons based on the model's "
        "principal reasons — not protected-class attributes."
    )
    immutable_features = CFPB_IMMUTABLE
    requires_plausibility = False
    plausibility_threshold = 0.0
    requires_diversity = False
    requires_actionable_reasons = True


class EUAIActFramework(RegulatoryFramework):
    """
    EU AI Act (High-Risk) — transparency, non-discrimination, human oversight.

    Constraints:
      - Sex and Title are immutable (protected attributes).
      - CFs must be plausible (score ≥ 0.2).
      - Diversity required.
      - Bias monitoring: flag any change to protected attributes.
    """

    name = "EU AI Act (High-Risk)"
    description = (
        "EU AI Act classifies socioeconomic allocation systems as high-risk, "
        "requiring technical robustness, non-discrimination, and auditability."
        " Protected attributes must never drive recourse recommendations."
    )
    immutable_features = EU_AI_IMMUTABLE
    requires_plausibility = True
    plausibility_threshold = 0.2
    requires_diversity = True
    requires_actionable_reasons = True

    def validate(
        self, cf: np.ndarray, original: np.ndarray
    ) -> ComplianceResult:
        result = super().validate(cf, original)
        sex_idx = _FEAT_IDX.get("Sex")
        if sex_idx is not None and not np.isclose(
            cf[sex_idx], original[sex_idx], atol=1e-6
        ):
            result.violations.append(
                "Bias flag: Sex altered (protected attribute)"
            )
            result.is_compliant = False
        return result


class UnconstrainedFramework(RegulatoryFramework):
    """Research baseline — no constraints, pure distance optimisation."""

    name = "Research Baseline"
    description = (
        "Unconstrained optimisation — no immutability, plausibility, or "
        "diversity requirements. Used as a baseline to measure the cost "
        "of compliance."
    )
    immutable_features = set()
    requires_plausibility = False
    plausibility_threshold = 0.0
    requires_diversity = False
    requires_actionable_reasons = False


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------
FRAMEWORKS: Dict[str, type] = {
    "gdpr": GDPRFramework,
    "cfpb": CFPBFramework,
    "eu_ai_act": EUAIActFramework,
    "unconstrained": UnconstrainedFramework,
}

FRAMEWORK_DISPLAY_NAMES: Dict[str, str] = {
    "gdpr": "GDPR Article 22",
    "cfpb": "CFPB Circular 2022-03",
    "eu_ai_act": "EU AI Act (High-Risk)",
    "unconstrained": "Research Baseline",
}


def get_framework(
    name: str, X_train: Optional[np.ndarray] = None
) -> RegulatoryFramework:
    """Instantiate a regulatory framework by key name."""
    cls = FRAMEWORKS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown framework '{name}'. "
            f"Choose from: {list(FRAMEWORKS.keys())}"
        )
    return cls(X_train=X_train)
