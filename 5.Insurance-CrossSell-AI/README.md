# 🛡️ Insurance Cross-Sell Prediction Platform

An end-to-end machine learning application that predicts which existing health insurance customers are likely to purchase vehicle insurance.

The project combines Machine Learning, Explainable AI, MLOps, and Generative AI to help sales teams identify high-potential customers, understand model predictions, and generate personalized outreach messages.

---

## Features

- Single customer cross-sell prediction
- Batch prediction using CSV upload
- XGBoost model with threshold tuning for imbalanced data
- SHAP-based model explainability
- MLflow experiment tracking and model versioning
- FastAPI inference API
- Docker support
- Gemini-powered personalized outreach message generation

---

## Model Performance

| Metric | Value |
|---------|------:|
| ROC-AUC | **0.8416** |
| Accuracy | **72.8%** |
| F1 Score (Positive Class) | **0.55** |
| Recall (Positive Class) | **0.81** |
| Optimized Threshold | **0.714** |

---


## Tech Stack

### Machine Learning
- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy

### Explainability & MLOps
- SHAP
- MLflow

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit
- Plotly

### Generative AI
- Gemini 2.5 Flash API

### Deployment
- Docker
- Docker Compose

---

## Architecture

```text
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
XGBoost Model
    │
    ├── Threshold Tuning
    ├── MLflow Tracking
    └── SHAP Explainability
    │
    ▼
FastAPI
    │
    ▼
Streamlit Dashboard
    ├── Single Prediction
    ├── Batch Prediction
    ├── Explainability & Insights
    └── Gemini Outreach Generator
```

---

## Running Locally

### Clone the repository

```bash
git clone <repository-url>
cd insurance-cross-sell-platform
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python dataset.py
python main.py
```

### Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

### Run the FastAPI service

```bash
docker-compose up --build
```

---

## Future Improvements

- Deploy the application on AWS
- Add CI/CD pipeline using GitHub Actions
- Add user authentication
- Support automated model retraining
- Add monitoring and logging for deployed models

---

## Author

**Tanvi Dedhia**