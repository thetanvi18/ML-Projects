import os
import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def train_with_mlflow():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Insurance Cross-Sell Prediction")

    with mlflow.start_run() as run:
        # ── 1. Load data ──────────────────────────────────────────────────
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # ── 2. Clean data ─────────────────────────────────────────────────
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # ── 3. Train model ────────────────────────────────────────────────
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")

        # ── 4. Evaluate model ─────────────────────────────────────────────
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, predictor.predict_with_threshold(X_test), output_dict=True)
        logging.info("Model evaluation completed successfully")

        # Save best threshold for inference
        import json
        os.makedirs('models', exist_ok=True)
        with open('models/threshold.json', 'w') as f:
            json.dump({'threshold': predictor.best_threshold}, f)
        logging.info(f"Best threshold saved: {predictor.best_threshold:.3f}")

        # ── 5. Generate SHAP plots ────────────────────────────────────────
        logging.info("Generating SHAP explainability plots...")
        trainer.generate_shap_plots(X_train, n_samples=1000)
        logging.info("SHAP plots saved to reports/")

        # ── 6. MLflow logging ─────────────────────────────────────────────
        mlflow.set_tag('developer', 'Tanvi')
        mlflow.set_tag('model', trainer.model_name)
        mlflow.set_tag('preprocessing', 'OneHotEncoder, StandardScaler, MinMaxScaler, SMOTE')
        mlflow.set_tag('domain', 'Insurance Cross-Sell')

        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        mlflow.log_metric('f1_score', report['weighted avg']['f1-score'])

        # Log SHAP plots as artifacts
        mlflow.log_artifact('reports/shap_summary_bar.png')
        mlflow.log_artifact('reports/shap_beeswarm.png')

        # Log & register model
        mlflow.sklearn.log_model(trainer.pipeline, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "insurance_crosssell_model")

        logging.info("MLflow tracking completed successfully")

        # ── 7. Print results ──────────────────────────────────────────────
        print("\n============= Model Evaluation Results ==============")
        print(f"Model:          {trainer.model_name}")
        print(f"Accuracy:       {accuracy:.4f}")
        print(f"ROC AUC Score:  {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        print(f"MLflow Run ID:  {run.info.run_id}")
        print("View MLflow UI: run `mlflow ui` then go to http://localhost:5000")


if __name__ == "__main__":
    train_with_mlflow()
