import json
import os
import sys
import sqlite3
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import mlflow
import mlflow.sklearn

from src.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASS_TO_IDX,
    DATE_COLUMN,
    IDX_TO_CLASS,
    NUMERICAL_FEATURES,
    SEVERITY_CLASSES,
    TARGET_COLUMN,
    build_preprocessor,
    chronological_split,
    extract_baseline_distribution,
)


class XGBoostPipelineWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper around XGBClassifier to support sample_weight during fit inside a Pipeline,
    and map integer prediction outputs to original string class names.
    """
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.classes_ = np.array(SEVERITY_CLASSES)
        self.model = xgb.XGBClassifier(**xgb_params)

    def fit(self, X, y):
        # Convert string classes to integer indices
        y_arr = np.array(y)
        y_idx = np.array([CLASS_TO_IDX[label] for label in y_arr])
        sample_weights = compute_sample_weight("balanced", y_idx)
        self.model.fit(X, y_idx, sample_weight=sample_weights)
        return self

    def predict(self, X):
        preds_idx = self.model.predict(X)
        return np.array([IDX_TO_CLASS[idx] for idx in preds_idx])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def load_training_data() -> pd.DataFrame:
    """Load analytical dataset from SQLite or CSV fallback."""
    db_path = os.path.join("data", "traffic_accidents.db")
    enriched_csv = os.path.join("data", "output", "accidents_enriched.csv")

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM accident_analytics;", conn)
            conn.close()
            if not df.empty:
                print(f"[Data Loader] Loaded {len(df):,} records from SQLite analytical warehouse ({db_path}).")
                return df
        except Exception as e:
            print(f"[Data Loader] SQLite read fallback: {e}")

    if os.path.exists(enriched_csv):
        df = pd.read_csv(enriched_csv)
        print(f"[Data Loader] Loaded {len(df):,} records from CSV ({enriched_csv}).")
        return df

    raise FileNotFoundError("No accident analytical data found. Please run 'python main.py' first.")


def plot_and_save_confusion_matrix(cm, classes, output_path, title="Confusion Matrix", normalize=False):
    plt.figure(figsize=(6, 5))
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.ylabel("True Severity")
    plt.xlabel("Predicted Severity")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_and_evaluate():
    print("=" * 65)
    print("[*] TRAFFIC ACCIDENT SEVERITY: END-TO-END MLOPS TRAINING PIPELINE")
    print("=" * 65)

    df = load_training_data()

    # Drop any rows missing target or essential features
    df = df.dropna(subset=[TARGET_COLUMN, DATE_COLUMN]).copy()
    df = df[df[TARGET_COLUMN].isin(SEVERITY_CLASSES)].copy()

    # 1. Chronological Train / Val / Test Split
    train_df, val_df, test_df = chronological_split(df, train_frac=0.70, val_frac=0.15)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET_COLUMN]

    # Save baseline distributions for drift detection
    baseline_stats = extract_baseline_distribution(train_df)
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "baseline_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)
    print("[*] Exported models/baseline_distribution.json for continuous drift monitoring.")

    # 2. Setup MLflow Tracking (SQLite backend for MLflow 3.x with full Model Registry support)
    os.makedirs("data", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
    experiment_name = "traffic_accident_severity"
    mlflow.set_experiment(experiment_name)

    artifacts_dir = os.path.join("models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 3. Model Candidates
    model_configs = {
        "Baseline_LogisticRegression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            "params": {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 1000},
        },
        "RandomForest_Balanced": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 12,
                "class_weight": "balanced",
                "min_samples_split": 10,
            },
        },
        "XGBoost_Balanced": {
            "model": XGBoostPipelineWrapper(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective="multi:softprob",
                eval_metric="mlogloss",
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softprob",
                "class_weight": "balanced_sample_weights",
            },
        },
    }

    results = {}
    fitted_pipelines = {}

    for model_name, cfg in model_configs.items():
        print(f"\n---> Training & Evaluating Candidate: {model_name}...")
        start_t = time.time()

        with mlflow.start_run(run_name=model_name):
            # Build unified inference pipeline
            preprocessor = build_preprocessor()
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", cfg["model"]),
            ])

            # Strict featurization: fit on training data only
            pipeline.fit(X_train, y_train)
            train_duration = time.time() - start_t

            # Validation Predictions
            y_val_pred = pipeline.predict(X_val)
            val_acc = float(accuracy_score(y_val, y_val_pred))
            val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
            val_weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
            val_macro_rec = float(recall_score(y_val, y_val_pred, average="macro", zero_division=0))
            val_macro_prec = float(precision_score(y_val, y_val_pred, average="macro", zero_division=0))

            # Per-class Recall (Crucial for Fatal & Serious)
            per_class_rec = recall_score(y_val, y_val_pred, labels=SEVERITY_CLASSES, average=None, zero_division=0)
            fatal_rec = float(per_class_rec[0])
            serious_rec = float(per_class_rec[1])
            slight_rec = float(per_class_rec[2])

            # False Negative Rates for High-Severity Collisions
            fatal_fnr = 1.0 - fatal_rec
            serious_fnr = 1.0 - serious_rec

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_val_pred, labels=SEVERITY_CLASSES)

            # Log to MLflow
            mlflow.log_params(cfg["params"])
            mlflow.log_param("split_strategy", "chronological")
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("val_samples", len(val_df))
            mlflow.log_param("test_samples", len(test_df))

            mlflow.log_metrics({
                "train_duration_sec": train_duration,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
                "val_macro_recall": val_macro_rec,
                "val_macro_precision": val_macro_prec,
                "val_fatal_recall": fatal_rec,
                "val_serious_recall": serious_rec,
                "val_slight_recall": slight_rec,
                "val_fatal_false_negative_rate": fatal_fnr,
                "val_serious_false_negative_rate": serious_fnr,
            })

            # Save and log confusion matrices
            cm_path = os.path.join(artifacts_dir, f"{model_name}_cm.png")
            cm_norm_path = os.path.join(artifacts_dir, f"{model_name}_cm_norm.png")
            plot_and_save_confusion_matrix(cm, SEVERITY_CLASSES, cm_path, f"{model_name} Confusion Matrix")
            plot_and_save_confusion_matrix(cm, SEVERITY_CLASSES, cm_norm_path, f"{model_name} Normalized CM", normalize=True)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(cm_norm_path)

            # Classification Report
            cls_rep = classification_report(y_val, y_val_pred, labels=SEVERITY_CLASSES, output_dict=True, zero_division=0)
            rep_path = os.path.join(artifacts_dir, f"{model_name}_report.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(cls_rep, f, indent=2)
            mlflow.log_artifact(rep_path)

            # Log Model Artifact using cloudpickle format
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_id = mlflow.active_run().info.run_id

            results[model_name] = {
                "run_id": run_id,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_macro_recall": val_macro_rec,
                "val_fatal_recall": fatal_rec,
                "val_serious_recall": serious_rec,
                "val_slight_recall": slight_rec,
                "fatal_fnr": fatal_fnr,
                "train_duration_sec": train_duration,
            }
            fitted_pipelines[model_name] = pipeline

            print(f"    Validation Macro-F1   : {val_macro_f1:.4f}")
            print(f"    Validation Accuracy   : {val_acc:.4f}")
            print(f"    Fatal Recall (Alert)  : {fatal_rec:.2%} (FNR: {fatal_fnr:.2%})")
            print(f"    Serious Recall        : {serious_rec:.2%}")
            print(f"    Slight Recall         : {slight_rec:.2%}")

    # 4. Champion Model Selection
    print("\n" + "=" * 65)
    print("[+] MODEL COMPARISON TABLE (Validation Performance):")
    print("=" * 65)
    comp_df = pd.DataFrame(results).T.sort_values(by="val_macro_f1", ascending=False)
    print(comp_df[["val_macro_f1", "val_fatal_recall", "val_serious_recall", "val_accuracy", "fatal_fnr"]].to_string())

    # Champion is selected based on highest Macro-F1 while ensuring high Fatal Recall
    champion_name = comp_df.index[0]
    champion_pipeline = fitted_pipelines[champion_name]
    champion_run_id = results[champion_name]["run_id"]
    print(f"\n[CHAMPION] SELECTED CHAMPION MODEL: {champion_name} (Run ID: {champion_run_id})")

    # 5. Final Evaluation of Champion on Unseen Chronological Test Set
    print("\n---> Evaluating Champion Model on Unseen Chronological TEST Set...")
    y_test_pred = champion_pipeline.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec = recall_score(y_test, y_test_pred, labels=SEVERITY_CLASSES, average=None, zero_division=0)
    test_fatal_rec = float(test_rec[0])
    test_serious_rec = float(test_rec[1])
    test_slight_rec = float(test_rec[2])

    test_cm = confusion_matrix(y_test, y_test_pred, labels=SEVERITY_CLASSES)
    test_cm_path = os.path.join(artifacts_dir, "champion_test_cm.png")
    plot_and_save_confusion_matrix(test_cm, SEVERITY_CLASSES, test_cm_path, f"{champion_name} TEST Set Confusion Matrix", normalize=True)

    print(f"  Test Accuracy         : {test_acc:.4f}")
    print(f"  Test Macro-F1         : {test_macro_f1:.4f}")
    print(f"  Test Fatal Recall     : {test_fatal_rec:.2%}")
    print(f"  Test Serious Recall   : {test_serious_rec:.2%}")
    print(f"  Test Slight Recall    : {test_slight_rec:.2%}")

    # Log Test metrics to MLflow champion run
    with mlflow.start_run(run_id=champion_run_id):
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "test_fatal_recall": test_fatal_rec,
            "test_serious_recall": test_serious_rec,
            "test_slight_recall": test_slight_rec,
        })
        mlflow.log_artifact(test_cm_path)

        # Register Champion Model in MLflow Registry
        model_uri = f"runs:/{champion_run_id}/model"
        registered_model_name = "traffic_accident_severity_model"
        try:
            reg_model = mlflow.register_model(model_uri, registered_model_name)
            print(f"[+] Model registered in MLflow Registry as '{registered_model_name}' (Version: {reg_model.version})")
        except Exception as reg_err:
            print(f"[!] Note: Model registry notice ({reg_err}). Model logged as MLflow artifact.")

    # 6. Export Serialized Production Pipeline
    pipeline_export_path = os.path.join("models", "accident_severity_pipeline.joblib")
    joblib.dump(champion_pipeline, pipeline_export_path)
    print(f"[+] Exported Champion Pipeline Artifact to: {pipeline_export_path}")

    # 7. Export Model Metadata Manifest
    metadata = {
        "model_name": champion_name,
        "algorithm": champion_pipeline.named_steps["classifier"].__class__.__name__,
        "model_version": "1.0.0",
        "registered_model_name": "traffic_accident_severity_model",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "mlflow_run_id": champion_run_id,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all": ALL_FEATURES,
        },
        "target_classes": SEVERITY_CLASSES,
        "validation_metrics": results[champion_name],
        "test_metrics": {
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "test_fatal_recall": test_fatal_rec,
            "test_serious_recall": test_serious_rec,
            "test_slight_recall": test_slight_rec,
        },
        "all_model_comparison": results,
    }
    with open(os.path.join("models", "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[+] Exported Model Metadata Manifest to: models/model_metadata.json")

    print("\n" + "=" * 65)
    print("[SUCCESS] MLOPS MODEL TRAINING & REGISTRY COMPLETE!")
    print("=" * 65 + "\n")
    return champion_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()
