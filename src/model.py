import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from xgboost import XGBClassifier

from src.features import engineer_features, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_training_data(train_df: pd.DataFrame):
    y = train_df['Survived']
    age_mean = train_df['Age'].mean()
    age_std = train_df['Age'].std()

    X_raw = engineer_features(train_df, train_age_mean=age_mean, train_age_std=age_std)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y, scaler, age_mean, age_std


def train_and_save(train_path: str, test_path: str = None):
    train_df, _ = load_data(train_path, test_path or train_path)

    X, y, scaler, age_mean, age_std = prepare_training_data(train_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(max_depth=4, n_estimators=100, criterion='gini', random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = round(metrics.accuracy_score(y_test, rf.predict(X_test)) * 100, 2)
    rf_f1 = round(metrics.f1_score(y_test, rf.predict(X_test), average='weighted') * 100, 2)
    rf_cv = round(cross_val_score(rf, X, y, cv=10, scoring='accuracy').mean() * 100, 2)

    # XGBoost
    xgb = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=200,
                         objective='binary:logistic', booster='gbtree',
                         eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_acc = round(metrics.accuracy_score(y_test, xgb.predict(X_test)) * 100, 2)
    xgb_f1 = round(metrics.f1_score(y_test, xgb.predict(X_test), average='weighted') * 100, 2)
    xgb_cv = round(cross_val_score(xgb, X, y, cv=10, scoring='accuracy').mean() * 100, 2)

    print(f"Random Forest  — Accuracy: {rf_acc}%  F1: {rf_f1}%  CV Accuracy: {rf_cv}%")
    print(f"XGBoost        — Accuracy: {xgb_acc}%  F1: {xgb_f1}%  CV Accuracy: {xgb_cv}%")

    # Select best model based on cross-validation accuracy
    best_model = xgb if xgb_cv >= rf_cv else rf
    best_name = "xgboost" if xgb_cv >= rf_cv else "random_forest"

    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump({"age_mean": age_mean, "age_std": age_std}, MODELS_DIR / "train_stats.pkl")
    joblib.dump(X_train, MODELS_DIR / "X_train.pkl")

    print(f"\nSaved: {best_name} as model.pkl")
    return best_model, scaler, X_train, y_train, X_test, y_test


def load_model():
    model = joblib.load(MODELS_DIR / "model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    stats = joblib.load(MODELS_DIR / "train_stats.pkl")
    X_train = joblib.load(MODELS_DIR / "X_train.pkl")
    return model, scaler, stats, X_train


def predict_passenger(raw_input: dict, model, scaler, stats: dict) -> dict:
    """
    raw_input: dict with keys matching raw CSV columns (before feature engineering).
    Returns survival probability and binary prediction.
    """
    df = pd.DataFrame([raw_input])
    X = engineer_features(df, train_age_mean=stats['age_mean'], train_age_std=stats['age_std'])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= 0.5)
    return {"survived": prediction, "probability": round(float(prob), 4), "features": X.iloc[0].to_dict()}


if __name__ == "__main__":
    import sys
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/test.csv"
    train_and_save(train_path, test_path)
