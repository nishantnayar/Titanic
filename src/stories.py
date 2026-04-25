import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings('ignore')

from src.features import engineer_features, FEATURE_COLUMNS
from src.explain import (
    build_explainer,
    shap_waterfall_plot,
    get_shap_values_for_instance,
    format_shap_narrative,
    generate_counterfactuals,
    format_counterfactual_narrative,
    FEATURE_LABELS,
)

IMG_DIR = Path(__file__).parent.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# Five archetypes chosen to represent distinct passenger experiences
ARCHETYPES = [
    {
        "id": 856,
        "slug": "young_mother_3rd_class",
        "name": "Leah Rosen (Mrs. Sam Aks)",
        "profile": "18-year-old woman, 3rd class, travelling with infant son",
        "context": (
            "Leah emigrated from Russia with her baby. "
            "3rd class women with children had a complex path to the lifeboats — "
            "lower decks, unfamiliar ship, but the 'women and children first' rule nominally in their favour."
        ),
    },
    {
        "id": 653,
        "slug": "young_man_3rd_class",
        "name": "Johannes Halvorsen Kalvik",
        "profile": "21-year-old man, 3rd class, travelling alone",
        "context": (
            "Johannes was a young Norwegian emigrant heading to America. "
            "Young men travelling alone in 3rd class represented the group with the lowest survival odds — "
            "last priority in evacuation, furthest from the lifeboats."
        ),
    },
    {
        "id": 468,
        "slug": "older_man_1st_class",
        "name": "Mr. John Montgomery Smart",
        "profile": "56-year-old man, 1st class, travelling alone",
        "context": (
            "A wealthy 1st class passenger who did not survive. "
            "This case challenges the assumption that class alone determined survival. "
            "Being male overrode the class advantage entirely."
        ),
    },
    {
        "id": 219,
        "slug": "woman_1st_class",
        "name": "Miss. Albina Bazzani",
        "profile": "32-year-old woman, 1st class, travelling alone",
        "context": (
            "Albina represented the highest-survival demographic — "
            "1st class woman, travelling alone, near the boat deck. "
            "The model is highly confident here, and the SHAP explanation shows why."
        ),
    },
    {
        "id": 737,
        "slug": "mother_large_family_3rd_class",
        "name": "Mrs. Margaret Ford",
        "profile": "48-year-old woman, 3rd class, travelling with husband and 3 children",
        "context": (
            "Margaret travelled with her entire family. "
            "Despite being female, the combination of 3rd class, large family, "
            "and the chaos of evacuation proved fatal. "
            "A stark illustration that gender was not the only variable."
        ),
    },
]


def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    stats = joblib.load("models/train_stats.pkl")
    X_train = joblib.load("models/X_train.pkl")
    explainer = build_explainer(model, X_train)
    return model, scaler, stats, X_train, explainer


def get_passenger_data(passenger_id: int, train_df: pd.DataFrame,
                       scaler, stats: dict) -> tuple:
    row = train_df[train_df['PassengerId'] == passenger_id].iloc[0]
    df_single = pd.DataFrame([row])
    X_raw = engineer_features(df_single,
                               train_age_mean=stats['age_mean'],
                               train_age_std=stats['age_std'])
    X_scaled = scaler.transform(X_raw)
    return row, X_scaled[0], X_raw.iloc[0]


def build_story(archetype: dict, model, scaler, stats, X_train,
                train_df: pd.DataFrame, explainer) -> dict:
    passenger_id = archetype['id']
    slug = archetype['slug']

    row, X_scaled, X_features = get_passenger_data(passenger_id, train_df, scaler, stats)

    actual_survived = int(row['Survived'])
    prob = model.predict_proba(X_scaled.reshape(1, -1))[0][1]
    predicted = int(prob >= 0.5)

    # SHAP waterfall
    waterfall_path = IMG_DIR / f"story_shap_{slug}.png"
    shap_waterfall_plot(explainer, X_scaled,
                        passenger_label=archetype['name'],
                        save=False)
    plt.savefig(waterfall_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

    # SHAP narrative
    shap_vals = get_shap_values_for_instance(explainer, X_scaled)
    shap_text = format_shap_narrative(shap_vals, top_n=3)

    # Counterfactuals (only meaningful if model prediction matches reality)
    cf_text = None
    cf_prob = None
    try:
        cfs_df = generate_counterfactuals(model, X_train, X_scaled, n_cfs=1)
        cf_row = cfs_df.iloc[0]
        cf_prob = model.predict_proba(cfs_df[FEATURE_COLUMNS].iloc[[0]].values)[0][1]
        cf_text = format_counterfactual_narrative(X_scaled, cf_row, prob, cf_prob)
    except Exception:
        cf_text = "No clear counterfactual found for this passenger."

    return {
        "id": passenger_id,
        "slug": slug,
        "name": archetype['name'],
        "profile": archetype['profile'],
        "context": archetype['context'],
        "actual_survived": actual_survived,
        "predicted_survived": predicted,
        "survival_probability": round(float(prob), 4),
        "model_correct": actual_survived == predicted,
        "shap_narrative": shap_text,
        "counterfactual_narrative": cf_text,
        "counterfactual_probability": round(float(cf_prob), 4) if cf_prob else None,
        "waterfall_path": str(waterfall_path),
    }


def print_story(story: dict):
    outcome = "SURVIVED" if story['actual_survived'] else "DID NOT SURVIVE"
    correct = "Correct" if story['model_correct'] else "Incorrect"

    print("=" * 60)
    print(f"  {story['name']}")
    print(f"  {story['profile']}")
    print("=" * 60)
    print(f"\nActual outcome:        {outcome}")
    print(f"Model prediction:      {story['survival_probability']:.0%} survival probability")
    print(f"Prediction accuracy:   {correct}\n")
    print(f"Context:\n  {story['context']}\n")
    print(f"What drove this prediction:\n  {story['shap_narrative']}\n")
    if story['counterfactual_narrative']:
        print(f"What would have changed the outcome:\n  {story['counterfactual_narrative']}")
    print()


def run_all_stories(save_summary: bool = True) -> list:
    model, scaler, stats, X_train, explainer = load_artifacts()
    train_df = pd.read_csv("data/train.csv")

    stories = []
    for archetype in ARCHETYPES:
        print(f"Building story: {archetype['name']}...")
        story = build_story(archetype, model, scaler, stats, X_train, train_df, explainer)
        stories.append(story)
        print_story(story)

    if save_summary:
        summary_cols = [
            'name', 'profile', 'actual_survived', 'predicted_survived',
            'survival_probability', 'model_correct',
            'shap_narrative', 'counterfactual_narrative', 'counterfactual_probability'
        ]
        summary_df = pd.DataFrame(stories)[summary_cols]
        summary_df.to_csv("data/stories_summary.csv", index=False)
        print("Saved: data/stories_summary.csv")

    return stories


if __name__ == "__main__":
    run_all_stories()
