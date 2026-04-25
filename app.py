import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.features import engineer_features, FEATURE_COLUMNS
from src.explain import (
    build_explainer,
    shap_summary_plot,
    shap_waterfall_plot,
    generate_counterfactuals,
    format_counterfactual_narrative,
    FEATURE_LABELS,
)
from src.stories import ARCHETYPES, build_story
from src.regulatory_framework import (
    FRAMEWORKS,
    FRAMEWORK_DISPLAY_NAMES,
    get_framework,
    UnconstrainedFramework,
)
from src.counterfactual_methods import CounterfactualEngine

warnings.filterwarnings("ignore")

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


@st.cache_resource
def load_cf_engine():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    X_train = joblib.load("models/X_train.pkl")
    return CounterfactualEngine(model, X_train, scaler)


@st.cache_data
def load_stories():
    model, scaler, stats, X_train, explainer = load_artifacts()
    train_df = pd.read_csv("data/train.csv")
    stories = []
    for archetype in ARCHETYPES:
        story = build_story(
            archetype, model, scaler, stats, X_train, train_df, explainer
        )
        stories.append(story)
    return stories


def load_audit_results():
    path = Path(__file__).parent / "data" / "regulatory_audit_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_stories_cf_cache():
    """Load pre-computed CF results for the 5 archetype passengers."""
    path = Path(__file__).parent / "data" / "stories_cf_cache.pkl"
    if path.exists():
        return joblib.load(path)
    return None


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

PORT_OPTIONS = {
    "Southampton (S)": "S",
    "Cherbourg (C)": "C",
    "Queenstown (Q)": "Q",
}

METHOD_COLORS = {
    "nearest_neighbor": "#636EFA",
    "nice": "#EF553B",
    "dice_genetic": "#00CC96",
    "dice_kdtree": "#AB63FA",
    "ocean": "#FFA15A",
    "feature_tweaking": "#19D3F3",
}

METRIC_DESCRIPTIONS = {
    "validity": "% of CFs that flip the prediction (target: 100%)",
    "proximity": "Mean L1 distance from original (lower = better)",
    "sparsity": "Mean # features changed per CF (lower = better)",
    "plausibility": "Mean IsolationForest score 0–1 (higher = better)",
    "actionability": "% respecting immutability constraints (target: 100%)",
    "diversity": "Mean pairwise distance across k=3 CFs (higher = better)",
}


def build_raw_input(sex, age, pclass, title, siblings, parents_children,
                    fare, deck_label, port_label) -> dict:
    cabin = DECK_TO_CABIN[deck_label]
    embarked = PORT_OPTIONS[port_label]
    return {
        "PassengerId": 9999,
        "Name": TITLE_TO_NAME[title],
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
    X_raw = engineer_features(
        df, train_age_mean=stats["age_mean"], train_age_std=stats["age_std"]
    )
    X_scaled = scaler.transform(X_raw)
    prob = model.predict_proba(X_scaled)[0][1]
    return X_scaled[0], round(float(prob), 4)


def survival_gauge(prob: float):
    pct = int(prob * 100)
    color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
    st.markdown(
        f"""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 4rem; font-weight: 700; color: {color};">
                {pct}%
            </div>
            <div style="font-size: 1.1rem; color: #888; margin-top: -8px;">
                survival probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_banner(prob: float):
    if prob >= 0.7:
        msg = "The model predicts you would have <strong>survived</strong>."
        color = "#2ecc71"
    elif prob >= 0.5:
        msg = (
            "The model predicts you would have "
            "<strong>narrowly survived</strong>."
        )
        color = "#f39c12"
    elif prob >= 0.3:
        msg = (
            "The model predicts you would likely "
            "<strong>not have survived</strong>."
        )
        color = "#e67e22"
    else:
        msg = (
            "The model predicts you would "
            "<strong>not have survived</strong>."
        )
        color = "#e74c3c"

    st.markdown(
        f'<div style="background:{color}22; border-left: 4px solid {color}; '
        f'padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">'
        f'{msg}</div>',
        unsafe_allow_html=True,
    )


def compliance_badge(is_compliant: bool):
    if is_compliant:
        return "🟢 Compliant"
    return "🔴 Non-compliant"


# ---------------------------------------------------------------------------
# Sidebar — passenger input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚢 Your Passenger Profile")
    st.caption(
        "Step into 1912. Fill in your profile and find out if you'd have made it."
    )
    st.caption(
        "The model was trained on real survival data from 712 passengers. "
        "Your prediction reflects patterns it learned — not chance."
    )
    st.divider()

    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    pclass = st.selectbox(
        "Ticket Class",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1st Class", 2: "2nd Class", 3: "3rd Class"
        }[x],
    )
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    st.divider()
    siblings = st.slider("Siblings / Spouses aboard", 0, 8, 0)
    parents_children = st.slider("Parents / Children aboard", 0, 6, 0)

    st.divider()
    fare = st.slider("Ticket Fare (GBP)", 0, 520, 30)
    deck = st.selectbox("Deck", list(DECK_TO_CABIN.keys()))
    port = st.selectbox("Port of Embarkation", list(PORT_OPTIONS.keys()))

    st.divider()
    run = st.button(
        "Predict my survival", type="primary", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Would You Have Survived the Titanic?")
st.caption(
    "Your seat. Your fate. Built with the same AI fairness tools banks use "
    "to explain loan rejections — applied to 1912. "
    "Explore your prediction, the reasoning behind it, and what the law says about it."
)

(
    tab_you,
    tab_shap,
    tab_cf,
    tab_compliance,
    tab_compare,
    tab_stories,
    tab_about,
) = st.tabs([
    "Your Prediction",
    "Why the Model Decided This",
    "What Would Have Changed It",
    "Regulatory Compliance Engine",
    "Method Comparison",
    "5 Real Passengers",
    "About This Project",
])

# ---------------------------------------------------------------------------
# Tab 1 — Prediction
# ---------------------------------------------------------------------------

with tab_you:
    if not run:
        st.info(
            "Set your passenger profile in the sidebar and click "
            "**Predict my survival**."
        )
    else:
        model, scaler, stats, X_train, explainer = load_artifacts()
        raw = build_raw_input(
            sex, age, pclass, title, siblings,
            parents_children, fare, deck, port,
        )

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
            "Model: Random Forest trained on 712 Titanic passengers · Accuracy: 80.4% · "
            "This is a statistical prediction based on historical patterns, not a historical fact."
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
        st.markdown(
            "Each factor either pulled you toward survival or away from it. "
            "The chart below shows which ones mattered most — and by how much."
        )
        st.caption(
            "Technical: SHAP waterfall plot. Each bar is one feature's contribution to the "
            "survival probability, relative to the model's average prediction across all passengers. "
            "Red = pushes probability up · Blue = pushes probability down."
        )

        fig = shap_waterfall_plot(
            explainer, X_scaled,
            passenger_label="Your passenger", save=False,
        )
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("What mattered most across all 712 passengers?")
        st.markdown(
            "This shows the bigger picture — which factors the model found most predictive "
            "across every passenger, not just yours."
        )
        fig_summary = shap_summary_plot(explainer, X_train, save=False)
        st.pyplot(fig_summary)
        plt.close()
        st.caption(
            "Technical: Global SHAP summary plot · Dot colour = feature value (high/low) · "
            "X-axis = mean impact on survival probability across all passengers."
        )

# ---------------------------------------------------------------------------
# Tab 3 — Counterfactuals (Nearest-Neighbour)
# ---------------------------------------------------------------------------

with tab_cf:
    if "X_scaled" not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        model, scaler, stats, X_train, explainer = load_artifacts()
        X_scaled = st.session_state["X_scaled"]
        prob = st.session_state["prob"]

        st.subheader("What would have changed your outcome?")
        st.markdown(
            "Small changes, different fate. Below are the minimum differences "
            "that would have flipped the model's prediction for you — your counterfactual escape plan."
        )
        st.caption(
            "Technical: Nearest Unlike Neighbours from the training set — real passengers whose profiles "
            "were just different enough for the model to predict the opposite outcome. "
            "This is the same counterfactual reasoning banks must provide when rejecting a loan application."
        )

        try:
            cfs_df = generate_counterfactuals(
                model, X_train, X_scaled, n_cfs=3
            )
            for i in range(len(cfs_df)):
                cf_row = cfs_df.iloc[i]
                cf_prob = model.predict_proba(
                    cfs_df[FEATURE_COLUMNS].iloc[[i]].values
                )[0][1]
                narrative = format_counterfactual_narrative(
                    X_scaled, cf_row, prob, cf_prob
                )
                direction = "survived" if cf_prob > prob else "did not survive"
                delta = int((cf_prob - prob) * 100)
                sign = "+" if delta > 0 else ""
                with st.expander(
                    f"Scenario {i+1} — probability {sign}{delta}pp "
                    f"to {cf_prob:.0%} ({direction})",
                    expanded=(i == 0),
                ):
                    st.code(narrative, language=None)
        except Exception as e:
            st.warning(f"Could not generate counterfactuals: {e}")


# ---------------------------------------------------------------------------
# Tab 4 — Regulatory Compliance Engine  (NEW)
# ---------------------------------------------------------------------------

with tab_compliance:
    st.subheader("Regulatory Compliance Engine")
    st.markdown(
        "Real AI regulations impose rules on what counts as a valid explanation. "
        "Select a framework below to see which methods hold up under legal scrutiny — and at what cost."
    )
    st.caption(
        "Technical: Each regulatory framework defines immutable features (attributes that cannot be used as recourse) "
        "and constraints on plausibility and diversity. The scorecard shows how all 6 CF methods perform against those constraints."
    )

    if "X_scaled" not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        X_scaled = st.session_state["X_scaled"]
        model, scaler, stats, X_train, explainer = load_artifacts()
        engine = load_cf_engine()

        fw_key = st.selectbox(
            "Regulatory Framework",
            options=list(FRAMEWORKS.keys()),
            format_func=lambda k: FRAMEWORK_DISPLAY_NAMES[k],
        )

        fw = get_framework(fw_key, X_train)

        st.markdown(f"**{fw.name}**")
        st.info(fw.description)

        immut_names = (
            sorted(fw.immutable_features) if fw.immutable_features else []
        )
        if immut_names:
            labels_str = ", ".join(
                f"`{FEATURE_LABELS.get(f, f)}`" for f in immut_names
            )
            st.markdown(f"**Immutable features:** {labels_str}")

        st.divider()
        st.markdown("#### Live Compliance Scorecard — All 6 Methods")

        # Cache key: unique per (passenger vector, framework). Avoids
        # re-running 12 CF generations when only the narrative changes.
        scorecard_key = f"scorecard_{hash(X_scaled.tobytes())}_{fw_key}"
        cost_key = f"cost_{hash(X_scaled.tobytes())}_{fw_key}"

        if scorecard_key not in st.session_state:
            with st.spinner("Generating counterfactuals for all 6 methods..."):
                rows = []
                orig_pred = int(model.predict(X_scaled.reshape(1, -1))[0])
                for method in CounterfactualEngine.METHOD_NAMES:
                    label = CounterfactualEngine.METHOD_LABELS[method]
                    try:
                        cfs = engine.generate(
                            X_scaled, method=method, framework=fw, n=3
                        )
                        if not cfs:
                            rows.append({
                                "Method": label,
                                "CFs generated": 0,
                                "Validity": "—",
                                "Actionability": "—",
                                "Plausibility": "—",
                                "Avg changes": "—",
                                "Avg distance": "—",
                                "Status": "⚠️ No CFs",
                            })
                            continue

                        validity = sum(
                            1 for cf in cfs if cf.prediction != orig_pred
                        ) / len(cfs)
                        actionability = sum(
                            1 for cf in cfs
                            if not any("Immutable" in v for v in cf.violations)
                        ) / len(cfs)
                        plaus = np.mean([cf.plausibility_score for cf in cfs])
                        avg_changes = np.mean([cf.n_changes for cf in cfs])
                        avg_dist = np.mean([cf.distance for cf in cfs])
                        all_compliant = all(cf.is_compliant for cf in cfs)
                        any_compliant = any(cf.is_compliant for cf in cfs)

                        if all_compliant:
                            status = "🟢 Fully compliant"
                        elif any_compliant:
                            status = "🟡 Partially compliant"
                        else:
                            status = "🔴 Non-compliant"

                        rows.append({
                            "Method": label,
                            "CFs generated": len(cfs),
                            "Validity": f"{validity:.0%}",
                            "Actionability": f"{actionability:.0%}",
                            "Plausibility": f"{plaus:.2f}",
                            "Avg changes": f"{avg_changes:.1f}",
                            "Avg distance": f"{avg_dist:.3f}",
                            "Status": status,
                        })
                    except Exception as exc:
                        rows.append({
                            "Method": label,
                            "CFs generated": 0,
                            "Validity": "—",
                            "Actionability": "—",
                            "Plausibility": "—",
                            "Avg changes": "—",
                            "Avg distance": "—",
                            "Status": f"❌ Error: {exc}",
                        })
            st.session_state[scorecard_key] = rows

        with st.expander("What do these columns mean?"):
            plain_descriptions = {
                "Validity": "Does this method actually flip the prediction? (target: 100%)",
                "Actionability": "Are the suggested changes legally usable under this framework? (target: 100%)",
                "Plausibility": "Do the suggested changes describe a realistic person? (higher = more realistic)",
                "Avg changes": "How many things need to change on average? (fewer = simpler explanation)",
                "Avg distance": "How different is the counterfactual from the original? (lower = closer to reality)",
            }
            for col, desc in plain_descriptions.items():
                st.markdown(f"**{col}** — {desc}")
            st.caption("Technical definitions: " + " · ".join(
                f"{k}: {v}" for k, v in METRIC_DESCRIPTIONS.items()
            ))

        scorecard_df = pd.DataFrame(st.session_state[scorecard_key])
        st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

        # ------ Cost-of-compliance comparison ------
        st.divider()
        st.markdown("#### The price of playing by the rules")
        st.markdown(
            "Without any legal constraints, an AI can flip your outcome by changing *anything* — "
            "even your age or gender. Once regulation steps in, those shortcuts are off-limits. "
            "The table below shows how many **extra profile changes** each method needs to make "
            "once it has to follow the selected framework's rules."
        )
        st.caption(
            "Technical: Unconstrained = no immutability or plausibility constraints (research baseline). "
            "Compliance cost (delta) = extra feature changes required under the selected framework vs. baseline. "
            "A delta of +2 means the method must change 2 additional features to find a valid explanation."
        )

        if cost_key not in st.session_state:
            with st.spinner("Computing compliance cost..."):
                cost_rows = []
                fw_unc = UnconstrainedFramework()
                for method in CounterfactualEngine.METHOD_NAMES:
                    label = CounterfactualEngine.METHOD_LABELS[method]
                    try:
                        cfs_unc = engine.generate(
                            X_scaled, method=method, framework=fw_unc, n=1
                        )
                        cfs_fw = engine.generate(
                            X_scaled, method=method, framework=fw, n=1
                        )
                        changes_unc = cfs_unc[0].n_changes if cfs_unc else None
                        changes_fw = cfs_fw[0].n_changes if cfs_fw else None
                        delta = (
                            (changes_fw - changes_unc)
                            if (changes_fw is not None and changes_unc is not None)
                            else None
                        )
                        cost_rows.append({
                            "Method": label,
                            "No rules: changes needed": changes_unc,
                            f"Under {fw.name}: changes needed": changes_fw,
                            "Extra changes due to regulation (↑ = more constrained)": (
                                f"+{delta}" if delta is not None and delta >= 0
                                else str(delta) if delta is not None
                                else "—"
                            ),
                        })
                    except Exception:
                        cost_rows.append({
                            "Method": label,
                            "No rules: changes needed": "—",
                            f"Under {fw.name}: changes needed": "—",
                            "Extra changes due to regulation (↑ = more constrained)": "—",
                        })
            st.session_state[cost_key] = cost_rows

        cost_df = pd.DataFrame(st.session_state[cost_key])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
        st.caption(
            "A delta of 0 means the method found a valid explanation without needing to use any "
            "legally restricted features. A positive delta means regulation forced it to find a "
            "longer, harder path. Methods with '—' could not find any compliant explanation at all."
        )

        # ------ Regulatory narrative ------
        st.divider()
        st.markdown("#### Regulatory Analysis")
        narratives = {
            "gdpr": (
                "**GDPR Article 22** prohibits fully automated decisions that "
                "significantly affect individuals unless specific safeguards "
                "exist. Counterfactual recourse must respect immutable personal "
                "attributes (Sex, Age, family structure). Methods that change "
                "these attributes produce explanations a Data Protection "
                "Authority could challenge. Only plausible, diverse CFs satisfy "
                "the 'meaningful information' standard. _Ref: GDPR Art. 22, "
                "Recital 71._"
            ),
            "cfpb": (
                "**CFPB Circular 2022-03** requires lenders using algorithmic "
                "models to provide specific principal reasons for adverse "
                "actions. Reasons must reflect the actual model drivers and "
                "must not cite protected-class attributes. DiCE and Feature "
                "Tweaking both excel here — they produce sparse, specific "
                "changes that map cleanly to human-readable adverse action "
                "notices. _Ref: CFPB Circular 2022-03, ECOA Regulation B._"
            ),
            "eu_ai_act": (
                "**EU AI Act (High-Risk)** requires that automated systems "
                "affecting safety or fundamental rights maintain technical "
                "robustness, non-discrimination, and human oversight. "
                "Counterfactuals citing Sex or Title as recourse drivers "
                "trigger a bias flag — the Act explicitly prohibits recourse "
                "paths that reinforce protected-class disadvantage. "
                "_Ref: EU AI Act Title III, Chapter 2._"
            ),
            "unconstrained": (
                "**Research Baseline** imposes no constraints. This is the "
                "gold standard for minimising distance and sparsity — but it "
                "has no legal standing. Comparing unconstrained metrics to "
                "regulated ones quantifies the **cost of compliance**: how much "
                "additional feature change is imposed by legal constraints."
            ),
        }
        st.markdown(narratives.get(fw_key, ""))


# ---------------------------------------------------------------------------
# Tab 5 — Method Comparison (NEW)
# ---------------------------------------------------------------------------

with tab_compare:
    st.subheader("Counterfactual Method Comparison")
    st.markdown(
        "Six algorithms. One goal: find the smallest change that flips the outcome. "
        "But they don't all get there the same way — and under regulation, some fail entirely."
    )
    st.caption(
        "Technical: 6 CF methods evaluated across 6 metrics (validity, proximity, sparsity, "
        "plausibility, actionability, diversity) on a held-out test slice, per regulatory framework."
    )

    audit_df = load_audit_results()

    if audit_df is None:
        st.warning(
            "No pre-computed audit results found. "
            "Run the following to generate them (takes ~2–5 min):\n\n"
            "```\npython -m src.compliance_evaluator\n```"
        )
    else:
        fw_filter = st.selectbox(
            "Filter by framework",
            options=["All"] + list(FRAMEWORKS.keys()),
            format_func=lambda k: (
                "All frameworks" if k == "All" else FRAMEWORK_DISPLAY_NAMES[k]
            ),
        )

        filtered = (
            audit_df
            if fw_filter == "All"
            else audit_df[audit_df["framework"] == fw_filter]
        )

        metrics = [
            "validity", "proximity", "sparsity",
            "plausibility", "actionability", "diversity",
        ]

        # ------ Radar chart ------
        st.markdown("#### How do the methods stack up overall?")
        st.markdown(
            "Each spoke of the chart is one quality you'd want in a fair AI explanation. "
            "A method that fills the whole shape is strong across the board."
        )
        st.caption(
            "Technical: Radar chart — 6 metrics normalised to [0, 1]. "
            "Proximity and sparsity are inverted (lower is better → higher on chart)."
        )

        pivot = (
            filtered.groupby("method_label")[metrics].mean().reset_index()
        )

        # Normalise
        norm = pivot.copy()
        for m in metrics:
            col = norm[m]
            mn, mx = col.min(), col.max()
            if mx > mn:
                norm[m] = (col - mn) / (mx - mn)
            else:
                norm[m] = 0.5
        # Invert proximity and sparsity: lower = better → flip so higher = better
        for inv in ["proximity", "sparsity"]:
            norm[inv] = 1.0 - norm[inv]

        fig_radar = go.Figure()
        for _, row in norm.iterrows():
            method_name = row["method_label"]
            values = [row[m] for m in metrics] + [row[metrics[0]]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=method_name,
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=480,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ------ Heatmap ------
        st.markdown("#### Method × Metric Heatmap")
        pivot_heat = (
            filtered.groupby("method_label")[metrics].mean().round(3)
        )
        fig_heat = px.imshow(
            pivot_heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            labels=dict(x="Metric", y="Method", color="Score"),
        )
        fig_heat.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        # ------ Pareto frontier ------
        st.markdown("#### The sweet spot: smallest change, fewest features altered")
        st.markdown(
            "Methods in the bottom-left corner give you the most minimal explanation — "
            "the closest counterfactual that changes the fewest things."
        )
        st.caption(
            "Technical: Pareto frontier — Proximity (L1 distance from original) vs. Sparsity "
            "(# features changed). Bottom-left dominates. Points on the frontier cannot improve "
            "on one metric without sacrificing the other."
        )
        pareto_data = filtered.groupby("method_label")[
            ["proximity", "sparsity"]
        ].mean().reset_index()
        fig_par = px.scatter(
            pareto_data,
            x="proximity",
            y="sparsity",
            text="method_label",
            size_max=12,
        )
        fig_par.update_traces(
            textposition="top center",
            marker=dict(size=12),
        )
        fig_par.update_layout(
            xaxis_title="Proximity (L1 distance — lower is better)",
            yaxis_title="Sparsity (# features changed — lower is better)",
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_par, use_container_width=True)

        # ------ Raw table ------
        st.markdown("#### Full Metrics Table")
        display_cols = ["method_label", "framework_label"] + metrics
        st.dataframe(
            filtered[display_cols]
            .rename(
                columns={
                    "method_label": "Method",
                    "framework_label": "Framework",
                }
            )
            .round(3),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 6 — 5 Real Passengers  (enhanced)
# ---------------------------------------------------------------------------

with tab_stories:
    st.subheader("Five passengers. Five different stories.")
    st.markdown(
        "These are real people from the Titanic manifest. Each one tells us something different "
        "about how survival was decided — and what the model learned from it."
    )
    st.caption(
        "Technical: 5 archetype passengers selected to represent distinct demographic intersections. "
        "All 6 counterfactual methods applied side-by-side per passenger, per regulatory framework."
    )

    with st.spinner("Loading passenger stories..."):
        stories = load_stories()

    model, scaler, stats, X_train, explainer = load_artifacts()
    train_df = pd.read_csv("data/train.csv")
    cf_cache = load_stories_cf_cache()

    fw_stories_key = st.selectbox(
        "Framework for counterfactual recourse",
        options=list(FRAMEWORKS.keys()),
        format_func=lambda k: FRAMEWORK_DISPLAY_NAMES[k],
        key="stories_fw",
    )

    # Only instantiate live engine + framework if cache is missing
    _live_engine = load_cf_engine() if cf_cache is None else None
    _fw_stories = get_framework(fw_stories_key, X_train) if cf_cache is None else None

    for story in stories:
        outcome_icon = "✓" if story["actual_survived"] else "✗"
        outcome_label = (
            "Survived" if story["actual_survived"] else "Did not survive"
        )
        correct_label = (
            "Model correct" if story["model_correct"] else "Model incorrect"
        )

        with st.expander(
            f"{outcome_icon}  {story['name']} — {story['profile']} "
            f"— {outcome_label}",
            expanded=False,
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**Actual outcome:** {outcome_label}")
                st.markdown(
                    f"**Model confidence:** "
                    f"{story['survival_probability']:.0%} survival probability"
                )
                st.markdown(f"**Prediction:** {correct_label}")
                st.markdown(f"\n_{story['context']}_")

                st.markdown("**What drove the prediction:**")
                for line in story["shap_narrative"].split("\n"):
                    st.markdown(f"- {line}")

            with col2:
                waterfall_path = Path(story["waterfall_path"])
                if waterfall_path.exists():
                    st.image(
                        str(waterfall_path),
                        caption=f"SHAP explanation — {story['name']}",
                    )

            # ------ All 6 CF methods side-by-side ------
            st.markdown("---")
            st.markdown(
                f"**All 6 counterfactual methods — "
                f"{FRAMEWORK_DISPLAY_NAMES[fw_stories_key]}**"
            )

            pid = story["id"]
            orig_pred = int(
                model.predict(
                    scaler.transform(
                        engineer_features(
                            pd.DataFrame([train_df[train_df["PassengerId"] == pid].iloc[0]]),
                            train_age_mean=stats["age_mean"],
                            train_age_std=stats["age_std"],
                        )
                    )
                )[0]
            )

            # Pull from pre-computed cache when available
            cached_fw = (
                cf_cache.get(pid, {}).get(fw_stories_key)
                if cf_cache is not None
                else None
            )

            method_results = []
            for method in CounterfactualEngine.METHOD_NAMES:
                label = CounterfactualEngine.METHOD_LABELS[method]
                try:
                    if cached_fw is not None:
                        cfs = cached_fw.get(method, [])
                    else:
                        # Fallback: compute live (first run before cache exists)
                        passenger_row = train_df[
                            train_df["PassengerId"] == pid
                        ].iloc[0]
                        df_single = pd.DataFrame([passenger_row])
                        X_raw = engineer_features(
                            df_single,
                            train_age_mean=stats["age_mean"],
                            train_age_std=stats["age_std"],
                        )
                        X_inst = scaler.transform(X_raw)[0]
                        cfs = _live_engine.generate(
                            X_inst, method=method, framework=_fw_stories, n=1
                        )

                    if cfs:
                        cf = cfs[0]
                        method_results.append({
                            "Method": label,
                            "Flips prediction": (
                                "✅" if cf.prediction != orig_pred else "❌"
                            ),
                            "# changes": cf.n_changes,
                            "Distance": f"{cf.distance:.3f}",
                            "Plausibility": f"{cf.plausibility_score:.2f}",
                            "Compliant": compliance_badge(cf.is_compliant),
                            "Violations": (
                                "; ".join(cf.violations)
                                if cf.violations
                                else "None"
                            ),
                        })
                    else:
                        method_results.append({
                            "Method": label,
                            "Flips prediction": "—",
                            "# changes": "—",
                            "Distance": "—",
                            "Plausibility": "—",
                            "Compliant": "⚠️ No CF",
                            "Violations": "—",
                        })
                except Exception as exc:
                    method_results.append({
                        "Method": label,
                        "Flips prediction": "—",
                        "# changes": "—",
                        "Distance": "—",
                        "Plausibility": "—",
                        "Compliant": "❌ Error",
                        "Violations": str(exc)[:60],
                    })

            st.dataframe(
                pd.DataFrame(method_results),
                use_container_width=True,
                hide_index=True,
            )

            # Narrative insight
            fw_narratives = {
                "gdpr": (
                    "Under GDPR, methods that change Sex or Age produce "
                    "non-compliant recourse. The compliance cost reveals "
                    "how many additional changes are legally required."
                ),
                "cfpb": (
                    "CFPB requires actionable reasons. Only methods that "
                    "suggest changes the passenger could realistically "
                    "have made satisfy the adverse action standard."
                ),
                "eu_ai_act": (
                    "EU AI Act flags any recourse path that involves "
                    "protected attributes. Bias-free methods provide "
                    "the only legally valid explanations here."
                ),
                "unconstrained": (
                    "Without constraints, all methods optimise purely "
                    "for distance and sparsity. These are the theoretical "
                    "minimum-cost counterfactuals — no legal standing."
                ),
            }
            st.caption(fw_narratives.get(fw_stories_key, ""))


# ---------------------------------------------------------------------------
# Tab 7 — About
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This started as a Kaggle dataset. It became a question about fairness in AI.

If a machine decides your fate — a loan, a job, a medical diagnosis — can it explain *why*?
And could that explanation hold up in court?

Most Titanic analyses stop at feature importance: which variables mattered most *on average*.
This project goes one level deeper: **why did the model make *this specific* decision,
and what would have changed it?**

That second question is called a **counterfactual explanation** — and it's the same reasoning
regulators require banks to use when rejecting a credit application.
The model must be able to say: *"Here is what would have changed the outcome."*

---

**Six Counterfactual Methods**

Each algorithm finds a different kind of "what if" — trading off simplicity, realism, and fairness.

| # | Method | What it does |
|---|--------|---------|
| 1 | Nearest Unlike Neighbour | Finds the closest real passenger with the opposite outcome |
| 2 | NICE | Optimises for simplicity, closeness, and realism simultaneously |
| 3 | DiCE-Genetic | Uses a genetic algorithm to generate diverse "what if" scenarios |
| 4 | DiCE-KDTree | Faster tree-search version — still finds diverse counterfactuals |
| 5 | OCEAN | Weights by how realistic the counterfactual person would be |
| 6 | Feature Tweaking | Follows the model's own decision paths for minimal tree-consistent changes |

**Three Regulatory Frameworks** — because not all explanations are legally equal

| Framework | What the law says you can't change |
|-----------|---------------|
| GDPR Article 22 | Sex, Age, and family structure are immutable; explanations must be plausible and diverse |
| CFPB Circular 2022-03 | Protected classes are off-limits; reasons must be specific and actionable |
| EU AI Act (High-Risk) | Protected attributes immutable; bias must be actively monitored |

**Six Evaluation Metrics** — how we measure whether an explanation is actually good

| Metric | Plain English | Technical definition |
|--------|--------------|---------------------|
| Validity | Does it actually flip the outcome? | % of CFs that change the prediction |
| Proximity | How different is it from you? | Mean L1 distance (lower = better) |
| Sparsity | How many things need to change? | Mean # features changed (lower = better) |
| Plausibility | Is the counterfactual a realistic person? | IsolationForest anomaly score 0–1 |
| Actionability | Is it legally usable? | % respecting immutability constraints |
| Diversity | Does it show multiple paths? | Mean pairwise distance across 3 CFs |

---

**Methodology**

- **Model:** Random Forest · 80.4% accuracy · 80.1% F1 on held-out test set
- **Explainability:** SHAP TreeExplainer — global and local feature attribution
- **Counterfactuals:** 6 methods stress-tested against 3 regulatory frameworks on a held-out test slice
- **Feature engineering:** Title extraction, deck mapping, age/fare binning, family size normalisation

**Stack:** Python · scikit-learn · SHAP · DiCE-ML · NICEx · Streamlit · Plotly

---

Built by [Nishant Nayar](https://linkedin.com/in/nishantnayar) · MS Analytics, University of Chicago
        """
    )
