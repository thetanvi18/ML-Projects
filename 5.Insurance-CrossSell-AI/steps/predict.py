import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_recall_curve, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)


class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['model']['store_path']
        self.pipeline = self.load_model()
        self.best_threshold = 0.5  # default, tuned during evaluation

    def load_config(self):
        import yaml
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        X = data.drop(columns=['Result'])
        y = data['Result']
        return X, y

    def find_best_threshold(self, X_val, y_val):
        """Find threshold that maximises F1 score for class 1."""
        y_proba = self.pipeline.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
        self.best_threshold = float(thresholds[best_idx])
        print(f"[INFO] Best threshold: {self.best_threshold:.3f} "
              f"(Precision={precisions[best_idx]:.2f}, "
              f"Recall={recalls[best_idx]:.2f}, "
              f"F1={f1_scores[best_idx]:.2f})")
        return self.best_threshold

    def predict_with_threshold(self, X, threshold=None):
        t = threshold if threshold is not None else self.best_threshold
        y_proba = self.pipeline.predict_proba(X)[:, 1]
        return (y_proba >= t).astype(int)

    def predict_proba_single(self, input_df):
        """Return prediction probability for a single row (FastAPI & Streamlit)."""
        proba = self.pipeline.predict_proba(input_df)[0]
        prob_cross_sell = float(proba[1])
        pred_class = int(prob_cross_sell >= self.best_threshold)
        confidence = prob_cross_sell if pred_class == 1 else (1 - prob_cross_sell)
        return pred_class, confidence, proba.tolist()

    def evaluate_model(self, X_test, y_test):
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        # Find best threshold first
        self.find_best_threshold(X_test, y_test)

        # Evaluate at default 0.5 threshold
        y_pred_default = (y_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_default)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Evaluate at best threshold
        y_pred_tuned = self.predict_with_threshold(X_test)
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        class_report_tuned = classification_report(y_test, y_pred_tuned)

        print(f"\n[Default threshold=0.50] Accuracy: {accuracy:.4f} | AUC-ROC: {roc_auc:.4f}")
        print(f"[Tuned  threshold={self.best_threshold:.2f}] Accuracy: {tuned_accuracy:.4f} | "
              f"F1(class 1): {f1_score(y_test, y_pred_tuned):.4f}")

        # Generate plots
        os.makedirs('reports', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_tuned)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Threshold={self.best_threshold:.2f})')
        plt.savefig('reports/confusion_matrix.png', bbox_inches='tight')
        plt.close()

        # ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title('ROC Curve')
        plt.savefig('reports/roc_curve.png', bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title('Precision-Recall Curve')
        plt.savefig('reports/pr_curve.png', bbox_inches='tight')
        plt.close()

        return tuned_accuracy, class_report_tuned, roc_auc
