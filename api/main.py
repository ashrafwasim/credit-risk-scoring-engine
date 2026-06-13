# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:11:33 2026

@author: DELL G3
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts default probability for loan applicants with SHAP explanations",
    version="1.0.0"
)

# ── Load Model and Features ────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "xgboost_credit_risk_v2.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_list.pkl")

print("Loading model...")
model        = joblib.load(MODEL_PATH)
feature_list = joblib.load(FEATURE_PATH)

print("Loading SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("API ready!")

# ── Input Schema ───────────────────────────────────────────
# These are the fields a user sends when requesting a risk score
# Optional fields will be filled with sensible defaults if missing
class LoanApplication(BaseModel):
    # Core loan details
    loan_amnt:    float = 10000
    term:         int   = 36
    installment:  float = 300

    # Borrower financials
    annual_inc:   float = 65000
    dti:          float = 15.0
    emp_length:   float = 5.0

    # Credit history
    fico_score:          float = 700
    credit_history_years: float = 10
    delinq_2yrs:         float = 0
    inq_last_6mths:      float = 0
    open_acc:            float = 8
    pub_rec:             float = 0
    revol_bal:           float = 5000
    revol_util:          float = 30.0
    total_acc:           float = 15
    mort_acc:            float = 0
    pub_rec_bankruptcies: float = 0

    # Account behaviour
    acc_open_past_24mths:  float = 3
    bc_open_to_buy:        float = 5000
    bc_util:               float = 40.0
    avg_cur_bal:           float = 8000
    tot_cur_bal:           float = 50000
    tot_coll_amt:          float = 0
    tot_hi_cred_lim:       float = 80000
    total_rev_hi_lim:      float = 20000
    total_bc_limit:        float = 15000
    total_bal_ex_mort:     float = 20000
    total_il_high_credit_limit: float = 20000

    # Categorical — home ownership
    home_ownership: str = "RENT"

    # Categorical — verification status
    verification_status: str = "Not Verified"

    # Categorical — loan purpose
    purpose: str = "debt_consolidation"


# ── Helper: Build Feature Vector ───────────────────────────
def build_feature_vector(app: LoanApplication) -> pd.DataFrame:
    """
    Takes a LoanApplication and builds a DataFrame
    with exactly the same columns the model was trained on.
    All missing features are filled with 0.
    """
    # Start with a row of zeros for all 76 features
    row = {feat: 0.0 for feat in feature_list}

    # Fill in the fields we received
    direct_fields = [
        'loan_amnt', 'term', 'installment', 'annual_inc', 'dti',
        'emp_length', 'fico_score', 'credit_history_years',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
        'pub_rec_bankruptcies', 'acc_open_past_24mths', 'bc_open_to_buy',
        'bc_util', 'avg_cur_bal', 'tot_cur_bal', 'tot_coll_amt',
        'tot_hi_cred_lim', 'total_rev_hi_lim', 'total_bc_limit',
        'total_bal_ex_mort', 'total_il_high_credit_limit'
    ]

    for field in direct_fields:
        if field in row:
            row[field] = getattr(app, field)

    # Handle one-hot encoded home ownership
    ownership_col = f"home_ownership_{app.home_ownership.upper()}"
    if ownership_col in row:
        row[ownership_col] = 1

    # Handle one-hot encoded verification status
    verification_col = f"verification_status_{app.verification_status}"
    if verification_col in row:
        row[verification_col] = 1

    # Handle one-hot encoded purpose
    purpose_col = f"purpose_{app.purpose.lower()}"
    if purpose_col in row:
        row[purpose_col] = 1

    return pd.DataFrame([row])[feature_list]


# ── Helper: Risk Tier ──────────────────────────────────────
def get_risk_tier(probability: float) -> dict:
    if probability < 0.15:
        return {
            "tier": "Low Risk",
            "recommendation": "Approve",
            "description": "Applicant shows strong repayment indicators"
        }
    elif probability < 0.30:
        return {
            "tier": "Medium-Low Risk",
            "recommendation": "Approve with standard review",
            "description": "Applicant shows acceptable risk profile"
        }
    elif probability < 0.50:
        return {
            "tier": "Medium Risk",
            "recommendation": "Manual underwriter review recommended",
            "description": "Applicant shows elevated risk indicators"
        }
    elif probability < 0.70:
        return {
            "tier": "High Risk",
            "recommendation": "Senior review or decline",
            "description": "Applicant shows multiple risk factors"
        }
    else:
        return {
            "tier": "Very High Risk",
            "recommendation": "Decline",
            "description": "Applicant shows severe risk indicators"
        }


# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms API is running"""
    return {
        "status": "running",
        "message": "Credit Risk Scoring API is live",
        "version": "1.0.0",
        "endpoints": {
            "health":  "GET  /",
            "predict": "POST /predict",
            "docs":    "GET  /docs"
        }
    }


@app.post("/predict")
def predict(application: LoanApplication):
    """
    Main endpoint — takes loan application details
    and returns default probability with SHAP explanation
    """
    try:
        # Step 1 — Build feature vector
        X = build_feature_vector(application)

        # Step 2 — Get default probability
        probability = float(model.predict_proba(X)[0][1])

        # Step 3 — Get risk tier and recommendation
        risk_info = get_risk_tier(probability)

        # Step 4 — Compute SHAP explanation
        shap_vals = explainer.shap_values(X)[0]

        # Step 5 — Get top 10 factors driving the decision
        shap_df = pd.DataFrame({
            "feature": feature_list,
            "value":   X.iloc[0].values,
            "impact":  shap_vals
        })

        # Positive impact = increases default risk
        # Negative impact = decreases default risk
        top_risk_factors = (
            shap_df[shap_df["impact"] > 0]
            .sort_values("impact", ascending=False)
            .head(5)
            .apply(lambda r: {
                "feature": r["feature"],
                "value":   round(float(r["value"]), 2),
                "impact":  round(float(r["impact"]), 4),
                "direction": "increases risk"
            }, axis=1)
            .tolist()
        )

        top_protective_factors = (
            shap_df[shap_df["impact"] < 0]
            .sort_values("impact", ascending=True)
            .head(5)
            .apply(lambda r: {
                "feature": r["feature"],
                "value":   round(float(r["value"]), 2),
                "impact":  round(float(r["impact"]), 4),
                "direction": "decreases risk"
            }, axis=1)
            .tolist()
        )

        # Step 6 — Return complete response
        return {
            "status": "success",
            "prediction": {
                "default_probability": round(probability, 4),
                "default_probability_pct": f"{probability*100:.1f}%",
                "risk_tier":        risk_info["tier"],
                "recommendation":   risk_info["recommendation"],
                "description":      risk_info["description"]
            },
            "explanation": {
                "top_risk_factors":       top_risk_factors,
                "top_protective_factors": top_protective_factors
            },
            "input_summary": {
                "loan_amount":    application.loan_amnt,
                "term_months":    application.term,
                "annual_income":  application.annual_inc,
                "dti":            application.dti,
                "fico_score":     application.fico_score
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))