import os
import joblib
import yaml
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()
        self.feature_names = None

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def create_pipeline(self):
        self.numeric_features = ['Age', 'RegionID', 'AnnualPremium']
        self.categorical_features = ['Gender', 'PastAccident']
        self.passthrough_features = ['HasDrivingLicense', 'Switch']

        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age', 'RegionID']),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Gender', 'PastAccident']),
            ('passthrough', 'passthrough', ['HasDrivingLicense', 'Switch']),
        ])

        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'XGBClassifier': XGBClassifier,
        }

        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        # Use sklearn Pipeline (no SMOTE — class_weight='balanced' handles imbalance)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])

        return pipeline

    def feature_target_separator(self, data):
        X = data.drop(columns=['Result'])
        y = data['Result']
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        # Store feature names after preprocessing for SHAP
        ohe_cols = list(
            self.pipeline.named_steps['preprocessor']
            .named_transformers_['onehot']
            .get_feature_names_out(['Gender', 'PastAccident'])
        )
        self.feature_names = ['AnnualPremium', 'Age', 'RegionID'] + ohe_cols + ['HasDrivingLicense', 'Switch']

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)

    def generate_shap_plots(self, X_train, n_samples=500):
        """Generate SHAP summary plot and save to reports/."""
        os.makedirs('reports', exist_ok=True)

        # Use sample for speed
        X_sample = X_train.sample(min(n_samples, len(X_train)), random_state=42)

        # Get preprocessed data (after preprocessor but before SMOTE/model)
        preprocessor = self.pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(X_sample)

        # Build explainer on the underlying model
        model = self.pipeline.named_steps['model']

        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict_proba, X_transformed)

        shap_values = explainer(X_transformed)

        # SHAP summary bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values,
            X_transformed,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.title('Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/shap_summary_bar.png', dpi=150, bbox_inches='tight')
        plt.close()

        # SHAP beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values,
            X_transformed,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Feature Impact (Beeswarm)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("[OK] SHAP plots saved to reports/")
        return self.feature_names
