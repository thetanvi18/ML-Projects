from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import numpy as np
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insurance Cross-Sell Prediction API",
    description=(
        "Predict which existing health insurance customers are likely to purchase "
        "vehicle insurance. Built with scikit-learn, MLflow, and SHAP."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & threshold ──────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

import json
THRESHOLD_PATH = "models/threshold.json"
THRESHOLD = 0.5
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, 'r') as f:
        THRESHOLD = json.load(f).get('threshold', 0.5)


# ── Schema ────────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    Gender: str = Field(..., example="Male", description="Male or Female")
    Age: int = Field(..., ge=18, le=100, example=35)
    HasDrivingLicense: int = Field(..., ge=0, le=1, example=1)
    RegionID: float = Field(..., ge=1, le=52, example=28.0)
    Switch: int = Field(..., ge=0, le=1, example=0, description="Previously insured: 0=No, 1=Yes")
    PastAccident: str = Field(..., example="Yes", description="Yes, No, or Unknown")
    AnnualPremium: float = Field(..., ge=1000, le=100000, example=35000.0)

    @validator('Gender')
    def gender_must_be_valid(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @validator('PastAccident')
    def accident_must_be_valid(cls, v):
        if v not in ['Yes', 'No', 'Unknown']:
            raise ValueError("PastAccident must be 'Yes', 'No', or 'Unknown'")
        return v


class PredictionResponse(BaseModel):
    predicted_class: int
    prediction_label: str
    confidence: float
    probability_cross_sell: float
    probability_no_cross_sell: float
    business_recommendation: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "service": "Insurance Cross-Sell Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first: python main.py")

    # Build input dataframe
    input_dict = customer.model_dump()
    df = pd.DataFrame([input_dict])

    try:
        proba = model.predict_proba(df)[0]
        prob_cross_sell = float(proba[1])
        prob_no_cross_sell = float(proba[0])
        pred_class = int(prob_cross_sell >= THRESHOLD)
        confidence = prob_cross_sell if pred_class == 1 else prob_no_cross_sell
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Business recommendation
    if pred_class == 1:
        label = "Likely to Purchase"
        recommendation = (
            f"High cross-sell potential ({prob_cross_sell:.1%} probability). "
            "Recommend proactive outreach with a vehicle insurance offer. "
            "Priority: HIGH."
        )
    else:
        label = "Unlikely to Purchase"
        recommendation = (
            f"Low cross-sell potential ({prob_cross_sell:.1%} probability). "
            "Consider nurturing campaigns or re-engage after policy milestone. "
            "Priority: LOW."
        )

    return PredictionResponse(
        predicted_class=pred_class,
        prediction_label=label,
        confidence=round(confidence, 4),
        probability_cross_sell=round(prob_cross_sell, 4),
        probability_no_cross_sell=round(prob_no_cross_sell, 4),
        business_recommendation=recommendation,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(customers: list[CustomerInput]):
    """Batch prediction endpoint — accepts up to 100 customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 records.")

    df = pd.DataFrame([c.model_dump() for c in customers])
    try:
        preds = model.predict(df).tolist()
        probas = model.predict_proba(df)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probas)):
        results.append({
            "index": i,
            "predicted_class": pred,
            "prediction_label": "Likely to Purchase" if pred == 1 else "Unlikely to Purchase",
            "probability_cross_sell": round(prob, 4),
        })
    return {"total": len(results), "predictions": results}
