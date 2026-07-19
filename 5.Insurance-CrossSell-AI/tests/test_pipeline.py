"""
Tests for the Insurance Cross-Sell Prediction API and ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Pipeline Tests ────────────────────────────────────────────────────────────
class TestDataCleaning:
    def setup_method(self):
        from steps.clean import Cleaner
        self.cleaner = Cleaner()

    def _make_sample_df(self, n=10):
        return pd.DataFrame({
            'id': range(n),
            'Gender': ['Male', 'Female'] * (n // 2),
            'Age': [35] * n,
            'HasDrivingLicense': [1] * n,
            'RegionID': [28.0] * n,
            'Switch': [0] * n,
            'VehicleAge': ['1-2 Year'] * n,
            'PastAccident': ['No'] * n,
            'AnnualPremium': ['£32,000.00'] * n,
            'SalesChannelID': [26] * n,
            'DaysSinceCreated': [150] * n,
            'Result': [0] * n,
        })

    def test_clean_drops_unwanted_columns(self):
        df = self._make_sample_df()
        cleaned = self.cleaner.clean_data(df)
        assert 'id' not in cleaned.columns
        assert 'SalesChannelID' not in cleaned.columns
        assert 'VehicleAge' not in cleaned.columns
        assert 'DaysSinceCreated' not in cleaned.columns

    def test_clean_converts_annual_premium(self):
        df = self._make_sample_df()
        cleaned = self.cleaner.clean_data(df)
        assert cleaned['AnnualPremium'].dtype in [np.float64, np.float32, float]

    def test_clean_no_null_values(self):
        df = self._make_sample_df()
        cleaned = self.cleaner.clean_data(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_keeps_result_column(self):
        df = self._make_sample_df()
        cleaned = self.cleaner.clean_data(df)
        assert 'Result' in cleaned.columns


class TestDataGeneration:
    def test_dataset_generates_files(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("dataset", "dataset.py")
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        dataset_module.generate_insurance_dataset(n_train=100, n_test=50)
        assert os.path.exists("data/train.csv")
        assert os.path.exists("data/test.csv")

    def test_dataset_has_correct_columns(self):
        df = pd.read_csv("data/train.csv")
        expected_cols = ['Gender', 'Age', 'HasDrivingLicense', 'RegionID',
                         'Switch', 'PastAccident', 'AnnualPremium', 'Result']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataset_has_binary_target(self):
        df = pd.read_csv("data/train.csv")
        assert set(df['Result'].unique()).issubset({0, 1})


class TestFastAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists("models/model.pkl"):
            pytest.skip("Model not trained yet. Run python main.py first.")
        from app import app
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "model_loaded" in response.json()

    def test_predict_high_risk_customer(self):
        payload = {
            "Gender": "Male",
            "Age": 45,
            "HasDrivingLicense": 1,
            "RegionID": 28.0,
            "Switch": 0,
            "PastAccident": "Yes",
            "AnnualPremium": 35000.0
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probability_cross_sell" in data
        assert "business_recommendation" in data
        assert data["predicted_class"] in [0, 1]

    def test_predict_invalid_gender(self):
        payload = {
            "Gender": "Unknown",
            "Age": 35,
            "HasDrivingLicense": 1,
            "RegionID": 28.0,
            "Switch": 0,
            "PastAccident": "No",
            "AnnualPremium": 30000.0
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_age_out_of_range(self):
        payload = {
            "Gender": "Male",
            "Age": 150,  # Too old
            "HasDrivingLicense": 1,
            "RegionID": 28.0,
            "Switch": 0,
            "PastAccident": "No",
            "AnnualPremium": 30000.0
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 422
