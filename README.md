# The Counterfactual Compliance Engine
### Algorithmic Recourse Meets Regulatory Reality

**Author: Nishant Nayar**

> *"Your loan was denied. Here's why — and here's what you'd need to change."*  
> *That second sentence? In 2026, it's not a courtesy. In regulated industries, it's the law.*

---

## What This Project Actually Does

Most Titanic survival classifiers stop at prediction accuracy. Some add SHAP
feature importance. A few include basic counterfactual "what-ifs."

**This project asks a harder question:** What if this model had to pass a
regulatory audit?

I stress-tested **six counterfactual explanation methods** against real-world
compliance frameworks (GDPR Article 22, CFPB Circular 2022-03, EU AI Act) to
answer: **which explainability techniques survive when "because the model said
so" isn't good enough?**

The result is a production-grade XAI system that doesn't just explain
predictions — it demonstrates **what algorithmic recourse looks like under
legal constraints**.

### Try It Live

```bash
pip install -r requirements.txt
pip install NICEx --no-deps
pip install oceanpy
streamlit run app.py
```

Open the **Regulatory Compliance** tab. Select "GDPR Article 22" as your
framework. Watch what happens when counterfactual methods must respect
immutability constraints, actionability requirements, and plausibility
thresholds.

---

## Why This Matters

In May 2022, the US Consumer Financial Protection Bureau issued
[Circular 2022-03](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/)
clarifying that **credit denials based on ML models MUST include specific,
accurate reasons** — even when the model is an ensemble that doesn't naturally
produce them.

The EU's GDPR Article 22 goes further: individuals have a **right to
explanation** for automated decisions that significantly affect them.

The technical solution: **counterfactual explanations** — showing someone not
just why they were denied, but what changes would flip the decision.

The implementation gap: **Which counterfactual method actually satisfies
regulators?** Nobody benchmarks this. Until now.

---

## Project Structure

```
Titanic/
├── data/
│   ├── Titanic-Dataset.csv              # Original 891 passengers
│   ├── train.csv                        # 712 passengers (80/20 split)
│   ├── test.csv                         # 179 passengers
│   ├── stories_summary.csv              # 5 passenger archetypes
│   └── regulatory_audit_results.csv     # Compliance scorecard (generated)
├── src/
│   ├── features.py                      # Feature engineering pipeline
│   ├── explain.py                       # SHAP + nearest-neighbour CF
│   ├── stories.py                       # Passenger archetypes
│   ├── counterfactual_methods.py        # 6 CF methods, unified API (NEW)
│   ├── regulatory_framework.py          # GDPR/CFPB/AI Act classes  (NEW)
│   └── compliance_evaluator.py          # Automated audit scoring   (NEW)
├── models/
│   ├── model.pkl                        # Trained RF (80.4% accuracy)
│   ├── scaler.pkl                       # StandardScaler
│   ├── train_stats.pkl                  # Age mean/std for imputation
│   ├── X_train.pkl                      # Scaled training features
│   └── isolation_forest.pkl             # Plausibility detector (generated)
├── img/                                 # SHAP waterfall plots
├── app.py                               # Streamlit application (7 tabs)
├── requirements.txt
└── README.md
```

---

## The App: Seven Tabs, One Story

### Tab 1: Your Prediction
Enter a custom passenger profile. Get an instant survival probability with
confidence gauge.

### Tab 2: Why the Model Decided This (SHAP)
SHAP waterfall chart — features pushing survival probability up or down from
the model baseline.

### Tab 3: What Would Have Changed It
Three nearest-unlike-neighbour counterfactuals with diversity filtering.

### Tab 4: Regulatory Compliance Engine ⭐ **[THE KEY TAB]**

1. **Select a framework:** GDPR Art. 22 · CFPB 2022-03 · EU AI Act · Baseline
2. **Live scorecard:** All 6 methods evaluated for your current passenger
3. **Cost-of-compliance table:** Unconstrained vs. constrained feature-change delta
4. **Regulatory narrative:** Per-framework legal analysis with citations

### Tab 5: Method Comparison
Pre-computed audit over 150 test passengers:
- Radar chart (6 metrics × 6 methods, normalised)
- Method × Metric heatmap
- Pareto frontier (proximity vs. sparsity)
- Full metrics table with framework filter

*Generate with:* `python -m src.compliance_evaluator` (~2-5 min)

### Tab 6: Real Passenger Stories (Enhanced)
Five historical passengers — all 6 CF methods side-by-side per archetype,
with framework-specific compliance badges and narrative insight.

### Tab 7: About
Methodology, stack, and connections to financial services.

---

## The Six Counterfactual Methods

| # | Method | Approach | Package |
|---|--------|---------|---------|
| 1 | Nearest Unlike Neighbour | Closest training point with opposite prediction | Custom |
| 2 | NICE | Optimises sparsity / proximity / plausibility / none | `NICEx --no-deps` |
| 3 | DiCE-Genetic | Genetic algorithm, diverse CFs via DPP kernel | `dice-ml>=0.12` |
| 4 | DiCE-KDTree | KD-tree retrieval, guarantees plausibility | `dice-ml>=0.12` |
| 5 | OCEAN | CP solver, plausibility-weighted NN fallback on Windows | `oceanpy` |
| 6 | Feature Tweaking | RF decision-path traversal, minimal tree-consistent CFs | Custom |

All expose a unified interface:

```python
from src.counterfactual_methods import CounterfactualEngine

engine = CounterfactualEngine(model, X_train, scaler)
cfs = engine.generate(
    instance,
    method='nice',
    framework=gdpr_framework,
    n=3
)
# Returns List[Counterfactual] with validity, distance, plausibility, compliance
```

---

## The Three Regulatory Frameworks

| Framework | Immutable features | Plausibility | Diversity |
|-----------|-------------------|-------------|-----------|
| GDPR Article 22 | Sex, Age, SibSp, Parch | Required (≥ 0.3) | Required |
| CFPB Circular 2022-03 | Sex | Not required | Optional |
| EU AI Act (High-Risk) | Sex, Title | Required (≥ 0.2) | Required |
| Research Baseline | None | Not required | Optional |

Plausibility is scored via **Isolation Forest** trained on the training set
(saved to `models/isolation_forest.pkl`).

---

## The Six Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Validity | % of CFs that flip the prediction | 100% |
| Proximity | Mean L1 distance to original | Lower is better |
| Sparsity | Mean # features changed | Lower is better |
| Plausibility | Mean IsolationForest score (0–1) | Higher is better |
| Actionability | % respecting immutability constraints | 100% |
| Diversity | Mean pairwise L1 distance across k=3 CFs | Higher is better |

---

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# CF method packages
pip install "dice-ml>=0.12"
pip install NICEx --no-deps      # --no-deps avoids pandas conflict
pip install oceanpy               # CP solver (Windows: falls back to NN)
```

### Generate the pre-computed audit results

```bash
python -m src.compliance_evaluator          # full audit (~2-5 min, 150 rows)
python -m src.compliance_evaluator --quick  # 20-row quick test
```

This creates `data/regulatory_audit_results.csv` and
`models/isolation_forest.pkl` — needed by Tab 5.

---

## Technical Details

**Model:** Random Forest (scikit-learn, 80.4% accuracy, 80.1% F1)  
**Feature engineering:** Title, Deck, Age bins, Fare bins, family-size features  
**SHAP:** TreeExplainer — local waterfall + global beeswarm  
**Plausibility:** IsolationForest (contamination=0.1) normalised to [0, 1]  
**Stack:** Python · scikit-learn · SHAP · DiCE-ML · NICEx · Plotly · Streamlit

---

## Connection to Financial Services

The Titanic dataset makes this methodology emotionally accessible. The same
SHAP + counterfactual compliance engine transfers directly to credit risk:

- **GDPR Article 22** → EU mortgage / credit card models
- **CFPB 2022-03** → US consumer lending adverse action notices
- **EU AI Act** → Any EU high-risk AI system (hiring, credit, insurance)

**Phase 2 (next project):** The identical pipeline applied to German Credit /
HELOC — with fairness metrics, a REST `/explain` endpoint, and a PDF audit
report per prediction.

---

## Citations

**Regulatory:**
- GDPR Article 22 — [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- CFPB Circular 2022-03 — [CFPB](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03)
- EU AI Act (2024) — [European Parliament](https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.html)

**Academic:**
- Wachter et al. (2018), "Counterfactual Explanations without Opening the Black Box," *Harvard JOLT* 31(2)
- Mothilal, Sharma & Tan (2020), "Explaining ML Classifiers through Diverse Counterfactual Explanations," *FAT\**
- Brughmans & Martens (2024), "NICE: Nearest Instance Counterfactual Explanations," *DMKD*
- Guidotti (2022), "Counterfactual explanations and how to find them," *DMKD*

---

## Author

**Nishant Nayar**  
MS Analytics, University of Chicago · MBA Finance, Punjabi University

**Portfolio:** [nishantnayar.vercel.app](https://nishantnayar.vercel.app)  
**LinkedIn:** [linkedin.com/in/nishantnayar](https://linkedin.com/in/nishantnayar)

---

*"Every other Titanic project stops at Tab 3. This one goes to Tab 7."*
