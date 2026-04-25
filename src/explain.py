import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

IMG_DIR = Path(__file__).parent.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

FEATURE_LABELS = {
    'Pclass': 'Passenger Class',
    'Age': 'Age Group',
    'SibSp': 'Siblings / Spouses',
    'Parch': 'Parents / Children',
    'Fare': 'Fare Band',
    'Embarked': 'Port of Embarkation',
    'Sex': 'Sex',
    'Title': 'Title',
    'Deck': 'Deck',
    'relatives': 'Total Relatives',
    'not_alone': 'Traveling Alone',
    'Age_Class': 'Age x Class',
    'Fare_Per_Person': 'Fare Per Person',
}


def _readable_labels():
    from src.features import FEATURE_COLUMNS
    return [FEATURE_LABELS.get(col, col) for col in FEATURE_COLUMNS]


def build_explainer(model, X_train: np.ndarray):
    explainer = shap.TreeExplainer(model)
    return explainer


def shap_summary_plot(explainer, X_train: np.ndarray, save: bool = True):
    labels = _readable_labels()
    shap_values = explainer.shap_values(X_train)

    # For binary classifiers shap_values is a list; take class 1
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(vals, X_train, feature_names=labels,
                      max_display=len(labels), plot_type="dot", show=False)
    plt.title("Feature Impact on Survival Prediction", fontsize=14, pad=12)
    fig.patch.set_alpha(0)
    for _ax in fig.axes:
        _ax.set_facecolor("none")
    plt.tight_layout()

    if save:
        path = IMG_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {path}")

    return fig


def shap_waterfall_plot(explainer, X_instance: np.ndarray,
                        passenger_label: str = "Passenger", save: bool = True):
    labels = _readable_labels()
    shap_values = explainer(X_instance.reshape(1, -1))

    # Random Forest returns shape (n_samples, n_features, n_classes) — take class 1
    sv = shap_values[0, :, 1] if shap_values.values.ndim == 3 else shap_values[0]

    # Inject readable feature names into the Explanation object
    sv.feature_names = labels

    n_features = len(sv.values)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv, max_display=min(13, n_features), show=False)
    plt.title(f"Why the Model Decided: {passenger_label}", fontsize=13, pad=12)
    fig.patch.set_alpha(0)
    for _ax in fig.axes:
        _ax.set_facecolor("none")
    plt.tight_layout()

    if save:
        safe_label = passenger_label.replace(" ", "_").lower()
        path = IMG_DIR / f"shap_waterfall_{safe_label}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {path}")

    return fig


def get_shap_values_for_instance(explainer, X_instance: np.ndarray) -> dict:
    """Returns a dict of feature -> shap_value (scalar) for a single passenger."""
    from src.features import FEATURE_COLUMNS
    shap_vals = explainer.shap_values(X_instance.reshape(1, -1))
    sv = np.array(shap_vals)
    if sv.ndim == 3:
        # shape (n_samples, n_features, n_classes) — take sample 0, class 1
        vals = sv[0, :, 1]
    elif isinstance(shap_vals, list):
        # older shap: list of [class0_array, class1_array]
        vals = np.array(shap_vals[1][0])
    else:
        vals = sv[0]
    return {col: float(v) for col, v in zip(FEATURE_COLUMNS, vals)}


def format_shap_narrative(shap_dict: dict, top_n: int = 3) -> str:
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for feat, val in sorted_features[:top_n]:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "increased" if val > 0 else "decreased"
        lines.append(f"{label} {direction} survival probability by {abs(val):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DiCE counterfactual explanations
# ---------------------------------------------------------------------------

# Features the passenger could realistically have changed
ACTIONABLE_FEATURES = ['Pclass', 'Embarked', 'Fare', 'Deck', 'relatives', 'not_alone', 'Fare_Per_Person']

# Human-readable decode maps for narrative formatting
PCLASS_DECODE = {1: "1st class", 2: "2nd class", 3: "3rd class"}
EMBARKED_DECODE = {0: "Southampton", 1: "Cherbourg", 2: "Queenstown"}
FARE_DECODE = {0: "under GBP 8", 1: "GBP 8-14", 2: "GBP 14-31", 3: "GBP 31-99", 4: "GBP 99-250", 5: "over GBP 250"}


def generate_counterfactuals(model, X_train_scaled: np.ndarray,
                             X_instance_scaled: np.ndarray, n_cfs: int = 3) -> pd.DataFrame:
    """
    Finds the n_cfs nearest training examples that the model predicts differently
    from the query instance. Guaranteed to find results if opposite-class examples exist.
    """
    from src.features import FEATURE_COLUMNS
    from sklearn.metrics.pairwise import euclidean_distances

    instance_pred = model.predict(X_instance_scaled.reshape(1, -1))[0]
    train_preds = model.predict(X_train_scaled)

    # Keep only training points with the opposite predicted class
    opposite_mask = train_preds != instance_pred
    opposite_X = X_train_scaled[opposite_mask]

    if len(opposite_X) == 0:
        raise ValueError("No training examples with opposite prediction found.")

    # Rank by distance to the query instance
    dists = euclidean_distances(X_instance_scaled.reshape(1, -1), opposite_X)[0]
    ranked_idx = np.argsort(dists)

    # Pick diverse counterfactuals — skip near-duplicates (within 0.05 of a prior pick)
    selected = []
    for idx in ranked_idx:
        candidate = opposite_X[idx]
        if all(np.linalg.norm(candidate - opposite_X[s]) > 0.05 for s in selected):
            selected.append(idx)
        if len(selected) == n_cfs:
            break

    chosen = opposite_X[selected]
    cfs_df = pd.DataFrame(chosen, columns=FEATURE_COLUMNS)
    cfs_df['Survived'] = model.predict(chosen)
    cfs_df['distance'] = dists[selected]
    return cfs_df


def format_counterfactual_narrative(original_scaled: np.ndarray,
                                    cf_row: pd.Series,
                                    original_prob: float,
                                    cf_prob: float) -> str:
    from src.features import FEATURE_COLUMNS
    original = dict(zip(FEATURE_COLUMNS, original_scaled))
    changes = []

    for feat in ACTIONABLE_FEATURES:
        orig_val = round(original.get(feat, 0), 4)
        cf_val = round(cf_row.get(feat, 0), 4)
        if abs(orig_val - cf_val) < 0.001:
            continue
        label = FEATURE_LABELS.get(feat, feat)

        if feat == 'Pclass':
            changes.append(
                f"{label}: {PCLASS_DECODE.get(round(orig_val * 2 + 1), orig_val)} "
                f"-> {PCLASS_DECODE.get(round(cf_val * 2 + 1), cf_val)}"
            )
        elif feat == 'Embarked':
            changes.append(
                f"{label}: {EMBARKED_DECODE.get(round(orig_val * 2), orig_val)} "
                f"-> {EMBARKED_DECODE.get(round(cf_val * 2), cf_val)}"
            )
        elif feat == 'Fare':
            changes.append(
                f"{label}: {FARE_DECODE.get(round(orig_val * 5), orig_val)} "
                f"-> {FARE_DECODE.get(round(cf_val * 5), cf_val)}"
            )
        else:
            changes.append(f"{label}: {orig_val:.2f} -> {cf_val:.2f}")

    if not changes:
        return "No actionable changes found for this counterfactual."

    change_text = "\n  ".join(changes)
    return (
        f"Survival probability: {original_prob:.0%} -> {cf_prob:.0%}\n\n"
        f"What would need to change:\n  {change_text}"
    )
