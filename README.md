# Titanic — Explainable AI Project
**Author: Nishant Nayar**

> "The model predicted you would not survive. I asked it what would have changed that."

---

## What This Project Does

Most Titanic analyses stop at feature importance — which variables mattered most across all passengers. This project goes one level deeper: **why did the model make this specific decision, and what would have changed it?**

That second question is called a **counterfactual explanation**. It is the same reasoning regulators require when a bank rejects a credit application. The Titanic dataset makes it emotionally accessible. The methodology transfers directly to financial services.

**Try it:** Run `streamlit run app.py` and enter your 1912 passenger profile.

---

## The App

Five tabs:

| Tab | What it shows |
|---|---|
| Your Prediction | Survival probability gauge for your custom passenger profile |
| Why the Model Decided This | SHAP waterfall — each feature's push toward or away from survival |
| What Would Have Changed It | Counterfactual scenarios — nearest passengers the model predicted differently |
| 5 Real Passengers | Five archetypes from the actual dataset with full SHAP and counterfactual explanations |
| About | Methodology and connection to financial services XAI |

---

## Project Structure

```
Titanic/
├── data/
│   ├── Titanic-Dataset.csv       # original 891-row dataset
│   ├── train.csv                 # 712 rows — stratified 80/20 split
│   ├── test.csv                  # 179 rows
│   └── stories_summary.csv       # 5 archetype narratives
├── src/
│   ├── features.py               # feature engineering pipeline
│   ├── model.py                  # model training and saving
│   ├── explain.py                # SHAP + counterfactual logic
│   └── stories.py                # 5 real passenger archetypes
├── models/                       # saved model artifacts
├── img/                          # SHAP plots
├── app.py                        # Streamlit application
├── requirements.txt
├── PLAN.md                       # project roadmap
└── Titanic.ipynb                 # exploratory analysis notebook
```

---

## Setup

**Using the conda environment (recommended):**

```bash
conda activate titanic
pip install -r requirements.txt
```

**Run the app:**

```bash
conda activate titanic
streamlit run app.py
```

**Retrain the model:**

```bash
python -m src.model data/train.csv data/test.csv
```

---

## Methodology

### Model

Random Forest Classifier — selected over XGBoost based on held-out test performance.

| Metric | Score |
|---|---|
| Accuracy | 80.4% |
| F1 (weighted) | 80.1% |
| Train / Test split | 80 / 20 stratified |

### Feature Engineering

| Feature | Approach |
|---|---|
| Deck | Extracted from Cabin letter, mapped to numeric |
| Title | Extracted from Name, rare titles grouped |
| Age | Binned into 7 groups |
| Fare | Binned into 6 groups |
| relatives | SibSp + Parch |
| not_alone | 1 if travelling alone |
| Age_Class | Age bin × Pclass interaction |
| Fare_Per_Person | Fare / (relatives + 1) |
| Sex, Embarked | Label encoded |

Missing values: Age imputed with random draw within mean ± std. Embarked filled with Southampton (most common). Cabin missing mapped to Unknown deck.

### Explainability — SHAP

SHAP TreeExplainer generates both global (summary) and local (per-passenger waterfall) feature attributions. Each bar on the waterfall shows one feature pushing the survival probability up or down from the model's baseline.

### Counterfactuals

Nearest-neighbour search across training examples predicted in the opposite class. Diversity filtering ensures the three scenarios shown are meaningfully different from each other. This approach is guaranteed to find results and runs in milliseconds — no generative model required.

---

## Five Passenger Archetypes

| Passenger | Profile | Outcome | Model |
|---|---|---|---|
| Leah Rosen (Mrs. Sam Aks) | 18yo woman, 3rd class, travelling with infant | Survived | 55% — correct |
| Johannes Halvorsen Kalvik | 21yo man, 3rd class, travelling alone | Did not survive | 10% — correct |
| Mr. John Montgomery Smart | 56yo man, 1st class, travelling alone | Did not survive | 18% — correct |
| Miss. Albina Bazzani | 32yo woman, 1st class, travelling alone | Survived | 95% — correct |
| Mrs. Margaret Ford | 48yo woman, 3rd class, 4 family members | Did not survive | 30% — correct |

---

## Connection to Financial Services

The SHAP + counterfactual pattern used here is the same one banks need to deploy under model explainability regulations. A credit rejection system must be able to answer: "What would this applicant need to change to receive a different outcome?"

This project is Phase 0 of a portfolio demonstrating that capability. The next project applies identical methodology to a credit risk dataset.

---

## Background — The RMS Titanic

RMS Titanic was a British passenger liner operated by the White Star Line. It sank on 15 April 1912 after striking an iceberg during its maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died.

*Source: [Wikipedia](https://en.wikipedia.org/wiki/Titanic)*

---

## Author

**Nishant Nayar** — Techno-functional leader at the intersection of Data Science, Technology, and Business.

MS Analytics, University of Chicago · MBA Finance, Punjabi University

[LinkedIn](https://linkedin.com/in/nishantnayar) · [Portfolio](https://nishantnayar.vercel.app)
