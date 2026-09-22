"""
Train the baseline resume-screening classifier.

Model choice: L2-regularized logistic regression.
Why logistic regression for this repo:
  1. It is the standard baseline in bias-audit / fairness literature -- the
     audit methodology is the point of this repo, not squeezing out the last
     point of AUC with a black box.
  2. Coefficients are directly interpretable, which matters when an auditor
     (or a recruiter, like me) has to explain *why* a candidate was screened
     out -- a real requirement under audit regimes.
  3. It trains in seconds on CPU and is fully deterministic.

The model sees ONLY non-protected features. Protected attributes (gender,
race_ethnicity, age_band) are held out for the audit in audit.py -- the
whole point is to show that bias can still leak in through proxies
(referral_flag) and through biased historical labels.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "applicants.csv")
ART_DIR = os.path.join(ROOT, "artifacts")

NUMERIC = ["years_experience", "skill_assessment_score", "referral_flag",
           "employment_gap_months"]
CATEGORICAL = ["education_level"]


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def train_model(df, sample_weight=None, seed=SEED):
    X = df[NUMERIC + CATEGORICAL]
    y = df["hire"].to_numpy()
    pipe = build_pipeline()
    if sample_weight is None:
        pipe.fit(X, y)
    else:
        pipe.fit(X, y, clf__sample_weight=sample_weight)
    return pipe


def main():
    df = pd.read_csv(DATA_CSV)
    os.makedirs(ART_DIR, exist_ok=True)

    # Stratified split so the test set preserves the label balance.
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["hire"]
    )
    train_df.to_csv(os.path.join(ART_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(ART_DIR, "test.csv"), index=False)

    model = train_model(train_df)
    joblib.dump(model, os.path.join(ART_DIR, "model.joblib"))

    proba = model.predict_proba(test_df[NUMERIC + CATEGORICAL])[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(test_df["hire"], pred)
    auc = roc_auc_score(test_df["hire"], proba)

    metrics = {
        "model": "LogisticRegression(C=1.0, max_iter=1000)",
        "features": NUMERIC + CATEGORICAL,
        "protected_attributes_excluded": ["gender", "race_ethnicity", "age_band"],
        "train_n": len(train_df),
        "test_n": len(test_df),
        "test_accuracy_at_0.5": round(float(acc), 4),
        "test_auc": round(float(auc), 4),
    }
    with open(os.path.join(ART_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_out = test_df.copy()
    test_out["score"] = proba
    test_out.to_csv(os.path.join(ART_DIR, "test_with_scores.csv"), index=False)

    # Interpretable coefficients for the report
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    coef_table = (
        pd.DataFrame({"feature": feature_names, "coef": coefs})
        .sort_values("coef", ascending=False)
    )
    print("Baseline logistic regression trained (protected attributes excluded).")
    print(f"  test accuracy@0.5 = {acc:.4f}, test AUC = {auc:.4f}")
    print("  top coefficients:")
    print(coef_table.head(8).to_string(index=False))
    print(f"Artifacts saved to {ART_DIR}/")


if __name__ == "__main__":
    main()
