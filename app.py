import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from pathlib import Path

from src.features import engineer_features, FEATURE_COLUMNS
from src.explain import (
    build_explainer,
    shap_waterfall_plot,
    get_shap_values_for_instance,
    generate_counterfactuals,
    format_counterfactual_narrative,
    FEATURE_LABELS,
)
from src.stories import ARCHETYPES, build_story

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Would You Have Survived the Titanic?",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    stats = joblib.load("models/train_stats.pkl")
    X_train = joblib.load("models/X_train.pkl")
    explainer = build_explainer(model, X_train)
    return model, scaler, stats, X_train, explainer


@st.cache_data
def load_stories():
    model, scaler, stats, X_train, explainer = load_artifacts()
    train_df = pd.read_csv("data/train.csv")
    stories = []
    for archetype in ARCHETYPES:
        story = build_story(archetype, model, scaler, stats, X_train, train_df, explainer)
        stories.append(story)
    return stories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TITLE_TO_NAME = {
    "Mr": "Mr. John Smith",
    "Mrs": "Mrs. Jane Smith",
    "Miss": "Miss. Jane Smith",
    "Master": "Master. James Smith",
    "Rare": "Dr. James Smith",
}

DECK_TO_CABIN = {
    "Unknown": "U0",
    "A (Top — 1st class)": "A1",
    "B (1st class)": "B1",
    "C (1st class)": "C1",
    "D (1st/2nd class)": "D1",
    "E (All classes)": "E1",
    "F (2nd/3rd class)": "F1",
    "G (3rd class)": "G1",
}

PORT_OPTIONS = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}


def build_raw_input(sex, age, pclass, title, siblings, parents_children,
                    fare, deck_label, port_label) -> dict:
    fake_name = f"{TITLE_TO_NAME[title]}"
    cabin = DECK_TO_CABIN[deck_label]
    embarked = PORT_OPTIONS[port_label]
    return {
        "PassengerId": 9999,
        "Name": fake_name,
        "Sex": sex.lower(),
        "Age": float(age),
        "Pclass": int(pclass),
        "SibSp": int(siblings),
        "Parch": int(parents_children),
        "Fare": float(fare),
        "Cabin": cabin,
        "Embarked": embarked,
        "Ticket": "UNKNOWN",
    }


def predict(raw_input, model, scaler, stats):
    df = pd.DataFrame([raw_input])
    X_raw = engineer_features(df, train_age_mean=stats["age_mean"], train_age_std=stats["age_std"])
    X_scaled = scaler.transform(X_raw)
    prob = model.predict_proba(X_scaled)[0][1]
    return X_scaled[0], round(float(prob), 4)


def survival_gauge(prob: float):
    pct = int(prob * 100)
    color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
    st.markdown(
        f"""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 4rem; font-weight: 700; color: {color};">{pct}%</div>
            <div style="font-size: 1.1rem; color: #888; margin-top: -8px;">survival probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_banner(prob: float):
    if prob >= 0.7:
        msg, color = "The model predicts you would have **survived**.", "#2ecc71"
    elif prob >= 0.5:
        msg, color = "The model predicts you would have **narrowly survived**.", "#f39c12"
    elif prob >= 0.3:
        msg, color = "The model predicts you would likely **not have survived**.", "#e67e22"
    else:
        msg, color = "The model predicts you would **not have survived**.", "#e74c3c"

    st.markdown(
        f'<div style="background:{color}22; border-left: 4px solid {color}; '
        f'padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">'
        f'{msg}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — passenger input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚢 Your Passenger Profile")
    st.caption("Build your 1912 passenger and see what the model predicts.")
    st.divider()

    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    pclass = st.selectbox("Ticket Class", [1, 2, 3],
                          format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    st.divider()
    siblings = st.slider("Siblings / Spouses aboard", 0, 8, 0)
    parents_children = st.slider("Parents / Children aboard", 0, 6, 0)

    st.divider()
    fare = st.slider("Ticket Fare (GBP)", 0, 520, 30)
    deck = st.selectbox("Deck", list(DECK_TO_CABIN.keys()))
    port = st.selectbox("Port of Embarkation", list(PORT_OPTIONS.keys()))

    st.divider()
    run = st.button("Predict my survival", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Would You Have Survived the Titanic?")
st.caption(
    "An explainable AI project — the same counterfactual reasoning "
    "used in credit decisions, applied to a 1912 passenger dataset."
)

tab_you, tab_shap, tab_cf, tab_stories, tab_about = st.tabs([
    "Your Prediction", "Why the Model Decided This", "What Would Have Changed It",
    "5 Real Passengers", "About This Project"
])

# ---------------------------------------------------------------------------
# Tab 1 — Prediction
# ---------------------------------------------------------------------------

with tab_you:
    if not run:
        st.info("Set your passenger profile in the sidebar and click **Predict my survival**.")
    else:
        model, scaler, stats, X_train, explainer = load_artifacts()
        raw = build_raw_input(sex, age, pclass, title, siblings, parents_children, fare, deck, port)

        try:
            X_scaled, prob = predict(raw, model, scaler, stats)
            st.session_state["X_scaled"] = X_scaled
            st.session_state["prob"] = prob
            st.session_state["raw"] = raw
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        col1, col2 = st.columns([1, 2])

        with col1:
            survival_gauge(prob)
            verdict_banner(prob)

        with col2:
            st.subheader("Your passenger profile")
            profile = {
                "Sex": sex,
                "Age": age,
                "Class": {1: "1st", 2: "2nd", 3: "3rd"}[pclass],
                "Title": title,
                "Siblings / Spouses": siblings,
                "Parents / Children": parents_children,
                "Fare": f"GBP {fare}",
                "Deck": deck.split(" ")[0],
                "Port": port.split(" ")[0],
            }
            for k, v in profile.items():
                st.markdown(f"**{k}:** {v}")

        st.caption(
            "The model is a Random Forest trained on 712 Titanic passengers. "
            "Accuracy: 80.4%. This is a prediction, not a historical fact."
        )

# ---------------------------------------------------------------------------
# Tab 2 — SHAP
# ---------------------------------------------------------------------------

with tab_shap:
    if "X_scaled" not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        model, scaler, stats, X_train, explainer = load_artifacts()
        X_scaled = st.session_state["X_scaled"]

        st.subheader("What drove this prediction?")
        st.caption(
            "Each bar shows one feature pushing the survival probability up (red) or down (blue). "
            "The starting point is the model's average prediction across all passengers."
        )

        fig = shap_waterfall_plot(explainer, X_scaled,
                                  passenger_label="Your passenger", save=False)
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Feature importance across all 712 passengers")
        img_path = Path("img/shap_summary.png")
        if img_path.exists():
            st.image(str(img_path), caption="Global SHAP summary — dot colour = feature value, x-axis = impact on survival")

# ---------------------------------------------------------------------------
# Tab 3 — Counterfactuals
# ---------------------------------------------------------------------------

with tab_cf:
    if "X_scaled" not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        model, scaler, stats, X_train, explainer = load_artifacts()
        X_scaled = st.session_state["X_scaled"]
        prob = st.session_state["prob"]

        st.subheader("What would have changed your outcome?")
        st.caption(
            "These are the nearest real passengers in the training data "
            "whose outcome the model predicted differently. "
            "Each shows the minimum set of differences that flips the result."
        )

        try:
            cfs_df = generate_counterfactuals(model, X_train, X_scaled, n_cfs=3)

            for i in range(len(cfs_df)):
                cf_row = cfs_df.iloc[i]
                cf_prob = model.predict_proba(
                    cfs_df[FEATURE_COLUMNS].iloc[[i]].values
                )[0][1]
                narrative = format_counterfactual_narrative(X_scaled, cf_row, prob, cf_prob)

                direction = "survived" if cf_prob > prob else "did not survive"
                delta = int((cf_prob - prob) * 100)
                sign = "+" if delta > 0 else ""

                with st.expander(
                    f"Scenario {i+1} — probability {sign}{delta}pp to {cf_prob:.0%} ({direction})",
                    expanded=(i == 0)
                ):
                    st.code(narrative, language=None)

        except Exception as e:
            st.warning(f"Could not generate counterfactuals: {e}")

        st.divider()
        st.markdown(
            "**Why does this matter beyond 1912?**  \n"
            "This is the same logic regulators require when a bank rejects a credit application. "
            "The model must be able to say: 'Here is what would have changed the outcome.' "
            "Counterfactual explanations make automated decisions auditable — "
            "and that is a legal requirement in financial services today."
        )

# ---------------------------------------------------------------------------
# Tab 4 — 5 Real Passengers
# ---------------------------------------------------------------------------

with tab_stories:
    st.subheader("Five passengers. Five different stories.")
    st.caption(
        "Real passengers from the dataset, each representing a different demographic. "
        "SHAP and counterfactual explanations applied to each."
    )

    with st.spinner("Loading passenger stories..."):
        stories = load_stories()

    for story in stories:
        outcome_icon = "✓" if story["actual_survived"] else "✗"
        outcome_label = "Survived" if story["actual_survived"] else "Did not survive"
        correct_label = "Model correct" if story["model_correct"] else "Model incorrect"

        with st.expander(
            f"{outcome_icon}  {story['name']} — {story['profile']} — {outcome_label}",
            expanded=False
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**Actual outcome:** {outcome_label}")
                st.markdown(f"**Model confidence:** {story['survival_probability']:.0%} survival probability")
                st.markdown(f"**Prediction:** {correct_label}")
                st.markdown(f"\n_{story['context']}_")

                st.markdown("**What drove the prediction:**")
                for line in story["shap_narrative"].split("\n"):
                    st.markdown(f"- {line}")

                if story["counterfactual_narrative"] and "No actionable" not in story["counterfactual_narrative"]:
                    st.markdown("**What would have changed the outcome:**")
                    st.code(story["counterfactual_narrative"], language=None)

            with col2:
                waterfall_path = Path(story["waterfall_path"])
                if waterfall_path.exists():
                    st.image(str(waterfall_path),
                             caption=f"SHAP explanation — {story['name']}")

# ---------------------------------------------------------------------------
# Tab 5 — About
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
The question this project answers is not new. But the way it answers it is.

Most Titanic analyses stop at feature importance — which variables mattered most across all passengers.
This project goes one level deeper: **why did the model make this specific decision, and what would have changed it?**

That second question is called a **counterfactual explanation**. It is the same reasoning that regulators
require banks to use when rejecting a credit application. The model must be able to say:
*"Here is what would have changed the outcome."*

The Titanic dataset makes this emotionally accessible. The methodology transfers directly to financial services.

---

**Methodology**

- Model: Random Forest (80.4% accuracy, 80.1% F1 on held-out test set)
- Explainability: SHAP TreeExplainer — global and local feature attribution
- Counterfactuals: Nearest-neighbor search in the training set among opposite-predicted passengers, with diversity filtering
- Feature engineering: Title extraction, deck mapping, age and fare binning, family size features

**Stack:** Python, scikit-learn, SHAP, Streamlit

**Connection to financial services**
The same SHAP + counterfactual pattern used here is the foundation for the next project in this portfolio:
a credit risk model with explainable rejection reasons — the kind of output a bank's model risk team
and regulators actually need to see.

---

Built by [Nishant Nayar](https://linkedin.com/in/nishantnayar) · MS Analytics, University of Chicago
        """
    )
