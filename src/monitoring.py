import json
import math
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SEVERITY_CLASSES,
)


def calculate_psi(baseline_dist: Dict[str, float], current_dist: Dict[str, float], epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions:
    PSI = sum((Current_i - Baseline_i) * ln(Current_i / Baseline_i))
    """
    all_keys = set(baseline_dist.keys()).union(set(current_dist.keys()))
    psi_total = 0.0

    for k in all_keys:
        b_val = max(baseline_dist.get(k, 0.0), epsilon)
        c_val = max(current_dist.get(k, 0.0), epsilon)
        psi_total += (c_val - b_val) * math.log(c_val / b_val)

    return float(psi_total)


class ModelMonitor:
    """
    Continuous MLOps monitoring engine:
    Tracks data quality, class drift, location drift, feature drift, latency, and false-negative rates.
    Persists inference events to SQLite warehouse to survive API and dashboard restarts.
    """

    def __init__(
        self,
        baseline_path: str = "models/baseline_distribution.json",
        metadata_path: str = "models/model_metadata.json",
        db_path: str = "data/traffic_accidents.db",
    ):
        self.baseline_path = os.path.abspath(os.path.join(PROJECT_ROOT, baseline_path))
        self.metadata_path = os.path.abspath(os.path.join(PROJECT_ROOT, metadata_path))
        self.db_path = os.path.abspath(os.path.join(PROJECT_ROOT, db_path))

        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                self.baseline = json.load(f)
        else:
            self.baseline = {
                "target_distribution": {"Slight": 0.738, "Serious": 0.248, "Fatal": 0.014},
                "numerical_stats": {},
                "categorical_distribution": {},
            }

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.max_buffer_size = 5000
        self.inference_history: List[Dict[str, Any]] = []

        # Initialize SQLite persistence table and load existing history
        self._init_db()
        self._load_recent_history()

    def _init_db(self):
        """Ensures the inference_logs table and WAL mode are initialized."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        predicted_severity TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        probabilities TEXT NOT NULL,
                        risk_alert_level TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        status TEXT NOT NULL,
                        features TEXT NOT NULL,
                        ground_truth_severity TEXT DEFAULT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_logs_timestamp ON inference_logs(timestamp);")
                conn.commit()
        except Exception as e:
            print(f"[ModelMonitor] Warning: Could not initialize inference_logs table: {e}")

    def _load_recent_history(self):
        """Loads recent records from database into memory upon startup."""
        if not os.path.exists(self.db_path):
            return
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, timestamp, created_at, model_version, model_name,
                           predicted_severity, confidence, probabilities,
                           risk_alert_level, latency_ms, status, features, ground_truth_severity
                    FROM inference_logs
                    ORDER BY id DESC
                    LIMIT ?;
                """, (self.max_buffer_size,))
                rows = cur.fetchall()
                self.inference_history = []
                for row in reversed(rows):
                    try:
                        feats = json.loads(row["features"]) if row["features"] else {}
                    except Exception:
                        feats = {}
                    try:
                        probs = json.loads(row["probabilities"]) if row["probabilities"] else {}
                    except Exception:
                        probs = {}
                    rec = {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "created_at": row["created_at"],
                        "model_version": row["model_version"],
                        "model_name": row["model_name"],
                        "predicted_severity": row["predicted_severity"],
                        "confidence": row["confidence"],
                        "probabilities": probs,
                        "risk_alert": row["risk_alert_level"],
                        "risk_alert_level": row["risk_alert_level"],
                        "latency_ms": row["latency_ms"],
                        "status": row["status"],
                        "ground_truth_severity": row["ground_truth_severity"],
                        **feats,
                    }
                    self.inference_history.append(rec)
        except Exception as e:
            print(f"[ModelMonitor] Warning: Failed loading historical inference logs: {e}")

    def log_inference_event(
        self,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        status: str = "SUCCESS",
        ground_truth: Optional[str] = None,
    ):
        """
        Persists a real-time inference request to the analytical database and sliding buffer.
        Records: timestamp, input features, predicted severity, probabilities, model version, status, latency.
        """
        ts = time.time()
        created_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
        model_version = str(prediction.get("model_version") or self.metadata.get("model_version", "1.0.0"))
        model_name = str(prediction.get("model_name") or self.metadata.get("model_name", "RandomForest_Balanced"))
        predicted_severity = str(prediction.get("predicted_severity", "Slight"))
        confidence = float(prediction.get("confidence", 0.0))
        probabilities = prediction.get("probabilities", {})
        risk_alert = str(prediction.get("risk_alert_level") or prediction.get("risk_alert") or "STANDARD_RISK")
        latency_ms = float(prediction.get("latency_ms", 0.0))

        # Filter features to serializable dict
        clean_features = {str(k): v for k, v in features.items() if not str(k).startswith("_")}

        # 1. Persist to SQLite
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")
                conn.execute("""
                    INSERT INTO inference_logs (
                        timestamp, created_at, model_version, model_name,
                        predicted_severity, confidence, probabilities,
                        risk_alert_level, latency_ms, status, features, ground_truth_severity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts,
                    created_at,
                    model_version,
                    model_name,
                    predicted_severity,
                    confidence,
                    json.dumps(probabilities),
                    risk_alert,
                    latency_ms,
                    status,
                    json.dumps(clean_features),
                    ground_truth,
                ))
                conn.commit()
        except Exception as e:
            print(f"[ModelMonitor] Database persistence error: {e}")

        # 2. Maintain sliding in-memory buffer
        mem_record = {
            "timestamp": ts,
            "created_at": created_at,
            "model_version": model_version,
            "model_name": model_name,
            "predicted_severity": predicted_severity,
            "confidence": confidence,
            "probabilities": probabilities,
            "risk_alert": risk_alert,
            "risk_alert_level": risk_alert,
            "latency_ms": latency_ms,
            "status": status,
            "ground_truth_severity": ground_truth,
            **clean_features,
        }
        self.inference_history.append(mem_record)
        if len(self.inference_history) > self.max_buffer_size:
            self.inference_history.pop(0)

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates input records against boundary domain rules and schemas.
        """
        total_rows = len(data)
        if total_rows == 0:
            return {"valid_rate": 1.0, "total": 0, "violations": {}}

        violations = {
            "missing_required_features": 0,
            "invalid_speed_limit": 0,
            "invalid_hour": 0,
            "negative_counts": 0,
            "out_of_bounds_gps": 0,
        }

        # Check missing features
        for f in ALL_FEATURES:
            if f not in data.columns or data[f].isna().any():
                violations["missing_required_features"] += int(data[f].isna().sum()) if f in data.columns else total_rows

        if "speed_limit" in data.columns:
            violations["invalid_speed_limit"] = int((~data["speed_limit"].isin([20, 30, 40, 50, 60, 70])).sum())

        if "accident_hour" in data.columns:
            violations["invalid_hour"] = int((~data["accident_hour"].between(0, 23)).sum())

        for count_col in ["number_of_vehicles", "number_of_casualties"]:
            if count_col in data.columns:
                violations["negative_counts"] += int((data[count_col] < 1).sum())

        if "latitude" in data.columns and "longitude" in data.columns:
            invalid_coords = ~(data["latitude"].between(49.0, 61.5) & data["longitude"].between(-8.5, 2.5))
            violations["out_of_bounds_gps"] = int(invalid_coords.sum())

        total_violations = sum(violations.values())
        pass_rate = max(0.0, 1.0 - (total_violations / (total_rows * len(violations))))

        return {
            "total_records_checked": total_rows,
            "data_quality_score": round(pass_rate * 100, 2),
            "status": "PASS" if pass_rate >= 0.95 else "WARNING" if pass_rate >= 0.85 else "FAIL",
            "violations": violations,
        }

    def assess_class_drift(self, recent_predictions: List[str]) -> Dict[str, Any]:
        """
        Evaluates PSI of predicted class distribution vs baseline training distribution.
        Requires at least 5 samples for a statistically meaningful calculation.
        """
        baseline_dist = self.baseline.get("target_distribution", {"Slight": 0.738, "Serious": 0.248, "Fatal": 0.014})

        if not recent_predictions:
            return {
                "psi": None,
                "status": "NO_DATA",
                "display_text": "No data yet",
                "interpretation": "Awaiting live API prediction requests to measure class drift.",
                "sample_count": 0,
                "current_distribution": {},
                "baseline_distribution": baseline_dist,
            }

        sample_count = len(recent_predictions)
        counts = pd.Series(recent_predictions).value_counts(normalize=True).to_dict()
        current_dist = {cls_name: float(counts.get(cls_name, 0.0)) for cls_name in SEVERITY_CLASSES}

        # Need minimum 5 samples for a statistically meaningful distribution comparison
        if sample_count < 5:
            return {
                "psi": None,
                "status": "INSUFFICIENT_DATA",
                "display_text": "Insufficient data",
                "interpretation": f"Insufficient sample size ({sample_count} / 5 records). Minimum 5 records required to assess drift.",
                "sample_count": sample_count,
                "current_distribution": current_dist,
                "baseline_distribution": baseline_dist,
            }

        psi = calculate_psi(baseline_dist, current_dist)
        rounded_psi = round(psi, 4)

        if psi < 0.10:
            status = "STABLE"
            interpretation = "No significant class distribution drift detected."
        elif psi < 0.20:
            status = "MODERATE_DRIFT"
            interpretation = "Moderate class distribution shift observed. Monitor incoming distribution."
        else:
            status = "SIGNIFICANT_DRIFT"
            interpretation = "Significant class drift detected (PSI >= 0.20). Model retraining recommended."

        return {
            "psi": rounded_psi,
            "status": status,
            "display_text": f"{rounded_psi:.4f}",
            "interpretation": interpretation,
            "sample_count": sample_count,
            "current_distribution": current_dist,
            "baseline_distribution": baseline_dist,
        }

    def assess_location_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Monitors shifts in collision coordinates and spatial centroids.
        UK Baseline Centroid: approx lat 52.8, lon -1.5
        """
        if "latitude" not in current_data.columns or "longitude" not in current_data.columns:
            return {"status": "NO_COORDINATES", "centroid_drift_km": 0.0}

        valid = current_data.dropna(subset=["latitude", "longitude"])
        if len(valid) == 0:
            return {"status": "NO_VALID_COORDINATES", "centroid_drift_km": 0.0}

        cur_lat = float(valid["latitude"].mean())
        cur_lon = float(valid["longitude"].mean())

        # Baseline UK collision centroid from training data
        base_lat = 52.85
        base_lon = -1.65

        # Haversine distance in km
        r_earth = 6371.0
        d_lat = math.radians(cur_lat - base_lat)
        d_lon = math.radians(cur_lon - base_lon)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(base_lat)) * math.cos(math.radians(cur_lat)) * math.sin(d_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        drift_km = r_earth * c

        status = "STABLE" if drift_km < 50.0 else "REGIONAL_SHIFT" if drift_km < 120.0 else "SIGNIFICANT_LOCATION_DRIFT"

        return {
            "current_centroid": {"latitude": round(cur_lat, 4), "longitude": round(cur_lon, 4)},
            "baseline_centroid": {"latitude": base_lat, "longitude": base_lon},
            "centroid_drift_km": round(drift_km, 2),
            "status": status,
        }

    def evaluate_retraining_criteria(
        self,
        class_psi: Optional[float] = None,
        fatal_fnr: Optional[float] = None,
        data_quality_score: Optional[float] = 100.0,
        sample_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluates formal MLOps Retraining Triggers:
        1. Class PSI >= 0.20 (only when class_psi is calculated from sufficient sample size)
        2. Fatal False Negative Rate >= 0.60 (only when actual labeled outcomes exist)
        3. Data Quality Score < 85.0%
        4. Fresh sample accumulation >= 10,000 records
        """
        triggers = []
        if class_psi is not None and class_psi >= 0.20:
            triggers.append(f"Significant Class Distribution Drift (PSI={class_psi:.4f} >= 0.20)")

        if fatal_fnr is not None and fatal_fnr >= 0.60:
            triggers.append(f"Excessive Fatal False-Negative Rate (FNR={fatal_fnr:.1%} >= 60.0%)")

        if data_quality_score is not None and data_quality_score < 85.0:
            triggers.append(f"Degraded Input Data Quality Score ({data_quality_score:.1f}% < 85.0%)")

        if sample_count >= 10000:
            triggers.append(f"Fresh Data Volume Milestone reached ({sample_count:,} records >= 10,000)")

        retrain_recommended = len(triggers) > 0

        return {
            "retrain_recommended": retrain_recommended,
            "status": "RETRAIN_TRIGGERED" if retrain_recommended else "OPERATIONAL_STABLE",
            "active_triggers": triggers,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "retraining_workflow": "dags/mlops_pipeline_dag.py (or python src/train.py)",
        }

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Compiles an overall operational health summary based on real persisted prediction records.
        Synchronizes across independent processes (FastAPI & Streamlit).
        """
        records = []
        try:
            if os.path.exists(self.db_path):
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, timestamp, created_at, model_version, model_name,
                               predicted_severity, confidence, probabilities,
                               risk_alert_level, latency_ms, status, features, ground_truth_severity
                        FROM inference_logs
                        WHERE status = 'SUCCESS'
                        ORDER BY id DESC
                        LIMIT 5000;
                    """)
                    records = cur.fetchall()
        except Exception as e:
            print(f"[ModelMonitor] Error reading inference logs from DB: {e}")

        # Fallback to in-memory history if DB returned no records but in-memory exists
        if not records and self.inference_history:
            records = [r for r in self.inference_history if r.get("status") == "SUCCESS"]
            is_dict_list = True
        else:
            is_dict_list = False

        total_inferences = len(records)
        baseline_dist = self.baseline.get("target_distribution", {"Slight": 0.738, "Serious": 0.248, "Fatal": 0.014})

        if total_inferences == 0:
            return {
                "total_inferences": 0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "class_drift": {
                    "psi": None,
                    "status": "NO_DATA",
                    "display_text": "No data yet",
                    "interpretation": "Awaiting live API prediction activity.",
                    "sample_count": 0,
                    "current_distribution": {},
                    "baseline_distribution": baseline_dist,
                },
                "data_quality": {
                    "data_quality_score": None,
                    "status": "NO_DATA",
                    "display_text": "No data yet",
                    "total_records_checked": 0,
                    "violations": {},
                },
                "fatal_fnr": {
                    "fatal_fnr": None,
                    "status": "AWAITING_LABELS",
                    "display_text": "Awaiting ground truth labels",
                    "labeled_samples": 0,
                },
                "retraining_decision": {
                    "retrain_recommended": False,
                    "status": "OPERATIONAL_STABLE",
                    "active_triggers": [],
                    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "retraining_workflow": "dags/mlops_pipeline_dag.py",
                },
                "latest_timestamp": None,
            }

        if is_dict_list:
            latencies = [float(r.get("latency_ms", 0.0)) for r in records]
            preds = [str(r.get("predicted_severity", "Slight")) for r in records]
            labeled_records = [r for r in records if r.get("ground_truth_severity")]
            latest_ts = records[-1].get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(records[-1]["timestamp"]))
            feature_dicts = []
            for r in records:
                feat = {k: v for k, v in r.items() if k in ALL_FEATURES}
                if feat:
                    feature_dicts.append(feat)
        else:
            latencies = [float(r["latency_ms"]) for r in records]
            preds = [str(r["predicted_severity"]) for r in records]
            labeled_records = [r for r in records if r["ground_truth_severity"] is not None]
            latest_ts = records[0]["created_at"]
            feature_dicts = []
            for r in records:
                try:
                    f = json.loads(r["features"]) if r["features"] else {}
                    feature_dicts.append(f)
                except Exception:
                    pass

        avg_latency = round(float(np.mean(latencies)), 2)
        p95_latency = round(float(np.percentile(latencies, 95)), 2)
        min_latency = round(float(np.min(latencies)), 2)
        max_latency = round(float(np.max(latencies)), 2)

        class_drift = self.assess_class_drift(preds)

        if feature_dicts:
            df_features = pd.DataFrame(feature_dicts)
            dq_result = self.validate_data_quality(df_features)
            dq_score = dq_result["data_quality_score"]
            dq_status = dq_result["status"]
            dq_display = f"{dq_score:.1f}%"
        else:
            dq_result = {"violations": {}, "total_records_checked": 0}
            dq_score = 100.0
            dq_status = "PASS"
            dq_display = "100.0%"

        data_quality_info = {
            "data_quality_score": dq_score,
            "status": dq_status,
            "display_text": dq_display,
            "total_records_checked": len(feature_dicts),
            "violations": dq_result.get("violations", {}),
        }

        # Fatal FNR calculated strictly from labeled samples
        if not labeled_records:
            fatal_fnr_info = {
                "fatal_fnr": None,
                "status": "AWAITING_LABELS",
                "display_text": "Awaiting ground truth labels",
                "labeled_samples": 0,
            }
            fatal_fnr_val = None
        else:
            fatal_labeled = [
                r for r in labeled_records
                if (r.get("ground_truth_severity") if is_dict_list else r["ground_truth_severity"]) == "Fatal"
            ]
            if not fatal_labeled:
                fatal_fnr_info = {
                    "fatal_fnr": 0.0,
                    "status": "PASS",
                    "display_text": "0.0% (0 fatal in labeled sample)",
                    "labeled_samples": len(labeled_records),
                }
                fatal_fnr_val = 0.0
            else:
                fn_count = sum(
                    1 for r in fatal_labeled
                    if (r.get("predicted_severity") if is_dict_list else r["predicted_severity"]) != "Fatal"
                )
                fnr = fn_count / len(fatal_labeled)
                fatal_fnr_val = round(fnr, 4)
                fatal_fnr_info = {
                    "fatal_fnr": fatal_fnr_val,
                    "status": "FAIL" if fatal_fnr_val >= 0.60 else "PASS",
                    "display_text": f"{fatal_fnr_val:.1%}",
                    "labeled_samples": len(labeled_records),
                    "fatal_samples": len(fatal_labeled),
                    "fatal_misses": fn_count,
                }

        retrain_decision = self.evaluate_retraining_criteria(
            class_psi=class_drift.get("psi"),
            fatal_fnr=fatal_fnr_val,
            data_quality_score=dq_score,
            sample_count=total_inferences,
        )

        return {
            "total_inferences": total_inferences,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "class_drift": class_drift,
            "data_quality": data_quality_info,
            "fatal_fnr": fatal_fnr_info,
            "retraining_decision": retrain_decision,
            "latest_timestamp": latest_ts,
        }

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the most recent recorded inference requests for dashboard display."""
        logs = []
        try:
            if os.path.exists(self.db_path):
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, timestamp, created_at, model_version, model_name,
                               predicted_severity, confidence, probabilities,
                               risk_alert_level, latency_ms, status, features, ground_truth_severity
                        FROM inference_logs
                        ORDER BY id DESC
                        LIMIT ?;
                    """, (limit,))
                    for row in cur.fetchall():
                        try:
                            f = json.loads(row["features"]) if row["features"] else {}
                        except Exception:
                            f = {}
                        try:
                            p = json.loads(row["probabilities"]) if row["probabilities"] else {}
                        except Exception:
                            p = {}
                        logs.append({
                            "id": row["id"],
                            "created_at": row["created_at"],
                            "predicted_severity": row["predicted_severity"],
                            "confidence": row["confidence"],
                            "probabilities": p,
                            "risk_alert_level": row["risk_alert_level"],
                            "latency_ms": row["latency_ms"],
                            "status": row["status"],
                            "model_version": row["model_version"],
                            "speed_limit": f.get("speed_limit", "-"),
                            "weather_condition": f.get("weather_condition", "-"),
                            "day_of_week": f.get("day_of_week", "-"),
                        })
        except Exception as e:
            print(f"[ModelMonitor] Error fetching recent logs: {e}")

        if not logs and self.inference_history:
            for r in reversed(self.inference_history[-limit:]):
                logs.append({
                    "id": r.get("id", "-"),
                    "created_at": r.get("created_at", "-"),
                    "predicted_severity": r.get("predicted_severity", "-"),
                    "confidence": r.get("confidence", 0.0),
                    "probabilities": r.get("probabilities", {}),
                    "risk_alert_level": r.get("risk_alert_level", r.get("risk_alert", "-")),
                    "latency_ms": r.get("latency_ms", 0.0),
                    "status": r.get("status", "SUCCESS"),
                    "model_version": r.get("model_version", "1.0.0"),
                    "speed_limit": r.get("speed_limit", "-"),
                    "weather_condition": r.get("weather_condition", "-"),
                    "day_of_week": r.get("day_of_week", "-"),
                })
        return logs


# Singleton monitor
_monitor_instance = None


def get_monitor() -> ModelMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance


if __name__ == "__main__":
    monitor = get_monitor()
    # Test with simulated batch
    simulated_preds = ["Slight"] * 70 + ["Serious"] * 25 + ["Fatal"] * 5
    drift_result = monitor.assess_class_drift(simulated_preds)
    print("\n--- Monitoring Drift Assessment ---")
    print(json.dumps(drift_result, indent=2))
    retrain_eval = monitor.evaluate_retraining_criteria(drift_result["psi"])
    print("\n--- Retraining Decision ---")
    print(json.dumps(retrain_eval, indent=2))
