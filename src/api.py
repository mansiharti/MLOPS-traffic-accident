import os
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features import ALL_FEATURES, SEVERITY_CLASSES
from src.monitoring import get_monitor
from src.predict import get_predictor

app = FastAPI(
    title="Traffic Accident Severity Prediction & MLOps API",
    description=(
        "Production inference and MLOps monitoring service for classifying UK traffic accident "
        "severity (Fatal, Serious, Slight) under class imbalance with real-time drift tracking."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---

class CollisionFeatures(BaseModel):
    accident_hour: int = Field(14, ge=0, le=23, description="Hour of collision (0 to 23)")
    day_of_week: str = Field("Friday", description="Day of week (Monday to Sunday)")
    speed_limit: int = Field(30, description="Posted speed limit in mph (e.g. 20, 30, 40, 50, 60, 70)")
    number_of_vehicles: int = Field(2, ge=1, description="Total vehicles involved in the collision")
    number_of_casualties: int = Field(1, ge=1, description="Total casualties involved")
    road_type: str = Field("Single carriageway", description="Carriageway type (Single carriageway, Dual carriageway, Roundabout, etc.)")
    light_conditions: str = Field("Daylight", description="Lighting conditions (Daylight, Darkness - lights lit, etc.)")
    road_surface_conditions: str = Field("Dry", description="Road surface moisture (Dry, Wet or damp, Snow, Frost or ice, etc.)")
    weather_condition: str = Field("Fine no high winds", description="Atmospheric weather at scene")
    temperature: float = Field(10.0, description="Ambient temperature in Celsius")
    precipitation: float = Field(0.0, ge=0.0, description="Liquid precipitation equivalent in mm")
    wind_speed: float = Field(12.0, ge=0.0, description="Surface wind speed in km/h")
    latitude: Optional[float] = Field(53.4808, description="WGS84 latitude")
    longitude: Optional[float] = Field(-2.2426, description="WGS84 longitude")
    local_authority: Optional[str] = Field("Manchester", description="Local administrative district name or ONS code")

    model_config = {
        "json_schema_extra": {
            "example": {
                "accident_hour": 22,
                "day_of_week": "Saturday",
                "speed_limit": 60,
                "number_of_vehicles": 3,
                "number_of_casualties": 2,
                "road_type": "Single carriageway",
                "light_conditions": "Darkness - no lighting",
                "road_surface_conditions": "Wet or damp",
                "weather_condition": "Raining with high winds",
                "temperature": 5.0,
                "precipitation": 2.5,
                "wind_speed": 35.0,
                "latitude": 53.7997,
                "longitude": -1.5491,
                "local_authority": "Leeds",
            }
        }
    }


class PredictionResult(BaseModel):
    predicted_severity: str
    confidence: float
    probabilities: Dict[str, float]
    risk_alert_level: str
    latency_ms: float
    model_version: str
    model_name: str


class BatchCollisionRequest(BaseModel):
    collisions: List[CollisionFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    batch_size: int
    total_latency_ms: float
    avg_latency_ms: float
    model_version: str


class GroundTruthAuditItem(BaseModel):
    features: CollisionFeatures
    ground_truth_severity: str


class GroundTruthAuditRequest(BaseModel):
    audit_samples: List[GroundTruthAuditItem]


# --- API Routes ---

@app.get("/", tags=["System"])
def root():
    predictor = get_predictor()
    return {
        "service": "Traffic Accident Severity Prediction & MLOps API",
        "status": "HEALTHY",
        "active_model": predictor.metadata.get("model_name", "RandomForest_Balanced"),
        "model_version": predictor.metadata.get("model_version", "1.0.0"),
        "documentation": "/docs",
        "healthcheck": "/health",
        "monitoring": "/monitoring/metrics",
    }


@app.get("/health", tags=["System"])
def health_check():
    predictor = get_predictor()
    return {
        "status": "UP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_loaded": predictor.pipeline is not None,
        "classes": predictor.classes_,
    }


@app.get("/model/metadata", tags=["Model Registry"])
def get_model_metadata():
    predictor = get_predictor()
    return predictor.metadata


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_severity(collision: CollisionFeatures):
    predictor = get_predictor()
    monitor = get_monitor()

    input_data = collision.model_dump()
    result = predictor.predict(input_data)

    # Persist inference event to real-time monitoring storage
    monitor.log_inference_event(
        features=input_data,
        prediction=result,
        status="SUCCESS",
    )

    return PredictionResult(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_severity_batch(batch: BatchCollisionRequest):
    predictor = get_predictor()
    monitor = get_monitor()

    records = [c.model_dump() for c in batch.collisions]
    if not records:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty batch.")

    result = predictor.predict_batch(records)

    # Persist each event in batch to monitoring storage with individual/average latency
    avg_lat = result.get("avg_latency_ms", 0.0)
    version = result.get("model_version", "1.0.0")
    m_name = result.get("model_name", "RandomForest_Balanced")

    for rec, pred in zip(records, result["predictions"]):
        event_pred = {
            **pred,
            "latency_ms": avg_lat,
            "model_version": version,
            "model_name": m_name,
        }
        monitor.log_inference_event(
            features=rec,
            prediction=event_pred,
            status="SUCCESS",
        )

    return BatchPredictionResponse(**result)


@app.get("/monitoring/metrics", tags=["MLOps Monitoring"])
def get_monitoring_metrics():
    monitor = get_monitor()
    return monitor.get_summary_metrics()


@app.post("/monitoring/evaluate", tags=["MLOps Monitoring"])
def evaluate_ground_truth(request: GroundTruthAuditRequest):
    """
    Evaluates model performance against an audited ground-truth sample.
    Calculates False-Negative Rates on high-severity classes and evaluates retraining criteria.
    """
    predictor = get_predictor()
    monitor = get_monitor()

    if not request.audit_samples:
        raise HTTPException(status_code=400, detail="Audit sample list cannot be empty.")

    records = [item.features.model_dump() for item in request.audit_samples]
    y_true = [item.ground_truth_severity for item in request.audit_samples]

    batch_res = predictor.predict(records)
    y_pred = [p["predicted_severity"] for p in batch_res["predictions"]]

    avg_lat = batch_res.get("avg_latency_ms", 0.0)
    version = batch_res.get("model_version", "1.0.0")
    m_name = batch_res.get("model_name", "RandomForest_Balanced")

    for item, pred in zip(request.audit_samples, batch_res["predictions"]):
        rec = item.features.model_dump()
        event_pred = {
            **pred,
            "latency_ms": avg_lat,
            "model_version": version,
            "model_name": m_name,
        }
        monitor.log_inference_event(
            features=rec,
            prediction=event_pred,
            status="SUCCESS",
            ground_truth=item.ground_truth_severity,
        )

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rec = recall_score(y_true, y_pred, labels=SEVERITY_CLASSES, average=None, zero_division=0)

    fatal_rec = float(rec[0])
    serious_rec = float(rec[1])
    slight_rec = float(rec[2])
    fatal_fnr = 1.0 - fatal_rec

    drift_report = monitor.assess_class_drift(y_pred)
    retrain_decision = monitor.evaluate_retraining_criteria(
        class_psi=drift_report.get("psi"),
        fatal_fnr=fatal_fnr,
        sample_count=len(request.audit_samples),
    )

    return {
        "sample_size": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "fatal_recall": round(fatal_rec, 4),
        "fatal_false_negative_rate": round(fatal_fnr, 4),
        "serious_recall": round(serious_rec, 4),
        "slight_recall": round(slight_rec, 4),
        "class_drift": drift_report,
        "retraining_decision": retrain_decision,
    }
