import os
import sys
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SEVERITY_CLASSES,
    build_preprocessor,
    chronological_split,
    extract_baseline_distribution,
)
from src.predict import get_predictor
from src.monitoring import calculate_psi, get_monitor
from src.api import app


class TestTrafficAccidentMLOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.monitor = get_monitor()
        cls.client = TestClient(app)

        cls.sample_record = {
            "accident_hour": 19,
            "day_of_week": "Friday",
            "speed_limit": 60,
            "number_of_vehicles": 3,
            "number_of_casualties": 2,
            "road_type": "Single carriageway",
            "light_conditions": "Darkness - no lighting",
            "road_surface_conditions": "Wet or damp",
            "weather_condition": "Raining with high winds",
            "temperature": 4.5,
            "precipitation": 2.0,
            "wind_speed": 30.0,
            "latitude": 53.4808,
            "longitude": -2.2426,
            "local_authority": "Manchester",
        }

    def test_01_feature_preprocessor_structure(self):
        preprocessor = build_preprocessor()
        self.assertIsNotNone(preprocessor)
        transformer_names = [t[0] for t in preprocessor.transformers]
        self.assertIn("num", transformer_names)
        self.assertIn("cat", transformer_names)

    def test_02_chronological_split_integrity(self):
        dummy_df = pd.DataFrame({
            "accident_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"],
            "accident_time": ["10:00", "11:00", "12:00", "13:00", "14:00"],
            "accident_severity": ["Slight", "Slight", "Serious", "Fatal", "Slight"],
        })
        train, val, test = chronological_split(dummy_df, 0.6, 0.2)
        self.assertLessEqual(train["accident_date"].max(), val["accident_date"].min())
        self.assertLessEqual(val["accident_date"].max(), test["accident_date"].min())

    def test_03_model_artifact_and_classes(self):
        self.assertIsNotNone(self.predictor.pipeline)
        for cls_name in ["Fatal", "Serious", "Slight"]:
            self.assertIn(cls_name, self.predictor.classes_)

    def test_04_single_inference_pipeline(self):
        res = self.predictor.predict(self.sample_record)
        self.assertIn("predicted_severity", res)
        self.assertIn(res["predicted_severity"], SEVERITY_CLASSES)
        self.assertIn("probabilities", res)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)
        self.assertIn("risk_alert_level", res)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_05_batch_inference(self):
        batch = [self.sample_record, self.sample_record]
        res = self.predictor.predict(batch)
        self.assertEqual(res["batch_size"], 2)
        self.assertEqual(len(res["predictions"]), 2)
        self.assertIn("avg_latency_ms", res)

    def test_06_data_quality_validation(self):
        df_valid = pd.DataFrame([self.sample_record])
        audit = self.monitor.validate_data_quality(df_valid)
        self.assertEqual(audit["status"], "PASS")
        self.assertGreaterEqual(audit["data_quality_score"], 95.0)

        # Invalid record test
        bad_record = self.sample_record.copy()
        bad_record["speed_limit"] = 150  # invalid speed limit
        bad_record["accident_hour"] = 28   # invalid hour
        bad_audit = self.monitor.validate_data_quality(pd.DataFrame([bad_record]))
        self.assertGreater(bad_audit["violations"]["invalid_speed_limit"], 0)
        self.assertGreater(bad_audit["violations"]["invalid_hour"], 0)

    def test_07_class_drift_psi_computation(self):
        # Stable baseline matching
        stable_preds = ["Slight"] * 74 + ["Serious"] * 24 + ["Fatal"] * 2
        stable_drift = self.monitor.assess_class_drift(stable_preds)
        self.assertEqual(stable_drift["status"], "STABLE")
        self.assertLess(stable_drift["psi"], 0.10)

        # Severe drift condition
        drifted_preds = ["Fatal"] * 60 + ["Serious"] * 30 + ["Slight"] * 10
        severe_drift = self.monitor.assess_class_drift(drifted_preds)
        self.assertEqual(severe_drift["status"], "SIGNIFICANT_DRIFT")
        self.assertGreaterEqual(severe_drift["psi"], 0.20)

    def test_08_location_drift_assessment(self):
        df_loc = pd.DataFrame([{"latitude": 52.85, "longitude": -1.65}])
        loc_res = self.monitor.assess_location_drift(df_loc)
        self.assertEqual(loc_res["status"], "STABLE")
        self.assertLess(loc_res["centroid_drift_km"], 10.0)

    def test_09_retraining_trigger_evaluation(self):
        # Healthy status
        healthy_eval = self.monitor.evaluate_retraining_criteria(class_psi=0.03, fatal_fnr=0.35)
        self.assertFalse(healthy_eval["retrain_recommended"])
        self.assertEqual(healthy_eval["status"], "OPERATIONAL_STABLE")

        # Triggered status on high drift
        drift_eval = self.monitor.evaluate_retraining_criteria(class_psi=0.25, fatal_fnr=0.35)
        self.assertTrue(drift_eval["retrain_recommended"])
        self.assertEqual(drift_eval["status"], "RETRAIN_TRIGGERED")

    def test_10_fastapi_endpoints(self):
        # Root
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["status"], "HEALTHY")

        # Health
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertTrue(r_health.json()["model_loaded"])

        # Model Metadata
        r_meta = self.client.get("/model/metadata")
        self.assertEqual(r_meta.status_code, 200)
        self.assertIn("registered_model_name", r_meta.json())

        # Prediction
        r_pred = self.client.post("/predict", json=self.sample_record)
        self.assertEqual(r_pred.status_code, 200)
        self.assertIn(r_pred.json()["predicted_severity"], SEVERITY_CLASSES)

        # Batch Prediction
        r_batch = self.client.post("/predict/batch", json={"collisions": [self.sample_record]})
        self.assertEqual(r_batch.status_code, 200)
        self.assertEqual(r_batch.json()["batch_size"], 1)

        # Monitoring Metrics
        r_mon = self.client.get("/monitoring/metrics")
        self.assertEqual(r_mon.status_code, 200)
        self.assertIn("class_drift", r_mon.json())

        # Ground Truth Evaluation
        audit_payload = {
            "audit_samples": [
                {"features": self.sample_record, "ground_truth_severity": "Serious"}
            ]
        }
        r_audit = self.client.post("/monitoring/evaluate", json=audit_payload)
        self.assertEqual(r_audit.status_code, 200)
        self.assertEqual(r_audit.json()["sample_size"], 1)


if __name__ == "__main__":
    unittest.main()

