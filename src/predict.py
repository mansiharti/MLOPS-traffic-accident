import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd

from src.features import ALL_FEATURES, SEVERITY_CLASSES


class SeverityPredictor:
    """
    Production inference engine for Traffic Accident Severity classification.
    Loads the serialized end-to-end ColumnTransformer + Champion Classifier pipeline.
    """

    def __init__(
        self,
        model_path: str = "models/accident_severity_pipeline.joblib",
        metadata_path: str = "models/model_metadata.json",
    ):
        self.model_path = os.path.abspath(model_path)
        self.metadata_path = os.path.abspath(metadata_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run 'python src/train.py' first."
            )

        self.pipeline = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "model_name": "RandomForest_Balanced",
                "model_version": "1.0.0",
                "algorithm": "RandomForestClassifier",
            }

        # Validate classes
        classifier = self.pipeline.named_steps["classifier"]
        self.classes_ = list(getattr(classifier, "classes_", SEVERITY_CLASSES))

    def _prepare_dataframe(self, record_or_records: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(record_or_records, pd.DataFrame):
            df = record_or_records.copy()
        elif isinstance(record_or_records, list):
            df = pd.DataFrame(record_or_records)
        elif isinstance(record_or_records, dict):
            df = pd.DataFrame([record_or_records])
        else:
            raise TypeError("Expected input as dict, list of dicts, or pandas DataFrame.")

        # Ensure all expected columns are present with intelligent defaults
        defaults = {
            "accident_hour": 14,
            "speed_limit": 30,
            "number_of_vehicles": 2,
            "number_of_casualties": 1,
            "temperature": 10.0,
            "precipitation": 0.0,
            "wind_speed": 12.0,
            "road_type": "Single carriageway",
            "light_conditions": "Daylight",
            "road_surface_conditions": "Dry",
            "weather_condition": "Fine no high winds",
            "day_of_week": "Friday",
        }

        for col, def_val in defaults.items():
            if col not in df.columns:
                df[col] = def_val
            else:
                df[col] = df[col].fillna(def_val)

        return df[ALL_FEATURES]

    def _determine_risk_level(self, probs: Dict[str, float]) -> str:
        p_fatal = probs.get("Fatal", 0.0)
        p_serious = probs.get("Serious", 0.0)

        # High-sensitivity alert thresholds to mitigate deadly false negatives
        if p_fatal >= 0.08 or (p_fatal + p_serious) >= 0.40:
            return "CRITICAL_ALERT"
        elif p_serious >= 0.25 or (p_fatal + p_serious) >= 0.30:
            return "ELEVATED_RISK"
        return "STANDARD_RISK"

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Inference for a single collision event."""
        start_time = time.perf_counter()
        df = self._prepare_dataframe(record)

        preds = self.pipeline.predict(df)
        probas = self.pipeline.predict_proba(df)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        pred_class = str(preds[0])
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes_, probas[0])
        }
        risk_level = self._determine_risk_level(prob_dict)

        return {
            "predicted_severity": pred_class,
            "confidence": round(float(np.max(probas[0])), 4),
            "probabilities": prob_dict,
            "risk_alert_level": risk_level,
            "latency_ms": round(latency_ms, 2),
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_name": self.metadata.get("model_name", "RandomForest_Balanced"),
        }

    def predict_batch(self, records: Union[List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, Any]:
        """Inference for multiple collision events."""
        start_time = time.perf_counter()
        df = self._prepare_dataframe(records)

        preds = self.pipeline.predict(df)
        probas = self.pipeline.predict_proba(df)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        results = []
        for i in range(len(df)):
            pred_class = str(preds[i])
            prob_dict = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self.classes_, probas[i])
            }
            risk_level = self._determine_risk_level(prob_dict)
            results.append({
                "predicted_severity": pred_class,
                "confidence": round(float(np.max(probas[i])), 4),
                "probabilities": prob_dict,
                "risk_alert_level": risk_level,
            })

        return {
            "predictions": results,
            "batch_size": len(df),
            "total_latency_ms": round(latency_ms, 2),
            "avg_latency_ms": round(latency_ms / max(len(df), 1), 3),
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_name": self.metadata.get("model_name", "RandomForest_Balanced"),
        }

    def predict(self, record_or_records: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes prediction with probabilities, risk categorization, and inference latency.
        Dispatches to predict_single for dict and predict_batch for list/DataFrame.
        """
        if isinstance(record_or_records, dict):
            return self.predict_single(record_or_records)
        return self.predict_batch(record_or_records)


# Singleton predictor instance for efficient in-memory re-use
_predictor_instance = None


def get_predictor() -> SeverityPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SeverityPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    sample_input = {
        "accident_hour": 23,
        "speed_limit": 60,
        "number_of_vehicles": 3,
        "number_of_casualties": 2,
        "road_type": "Single carriageway",
        "light_conditions": "Darkness - no lighting",
        "road_surface_conditions": "Wet or damp",
        "weather_condition": "Raining with high winds",
        "day_of_week": "Saturday",
        "temperature": 4.5,
        "precipitation": 1.5,
        "wind_speed": 28.0,
    }
    result = predictor.predict(sample_input)
    print("\n--- Test Single Inference Output ---")
    print(json.dumps(result, indent=2))
