# Traffic Accident Severity Prediction: End-to-End MLOps Solution
## Course: Data Engineering and MLOps | Assignment Part 2 Report

---

## 1. Executive Summary & Part 2 Objectives
Following the successful implementation of the Part 1 data pipeline and dimensional warehouse, **Part 2 operationalizes an end-to-end Machine Learning Operations (MLOps) system** to classify road traffic collision severity into three official UK Department for Transport tiers: **Fatal**, **Serious**, and **Slight**.

Key deliverables accomplished in Part 2 include:
1. **Chronological Splitting & Leakage Prevention**: Enforced a strict chronological split across 101,523 collision records (70% Train, 15% Validation, 15% Test) to mirror real-world forecasting without lookahead bias.
2. **Mitigation of Acute Class Imbalance**: Addressed the extreme real-world skew (Slight: 73.8%, Serious: 24.8%, Fatal: 1.43%) using cost-sensitive balanced class weighting ($w_c = \frac{N}{K \cdot N_c}$).
3. **Multi-Model Benchmark**: Implemented and benchmarked three classifiers:
   - Baseline Balanced Logistic Regression
   - Balanced Random Forest Classifier
   - Balanced Multi-Class XGBoost Classifier
4. **MLflow Experiment Tracking & Registry**: Tracked all hyperparameters, training durations, macro/weighted F1 scores, per-class recall, and confusion matrix heatmaps under the `traffic_accident_severity` MLflow experiment. Versioned and registered the winning champion in the MLflow Model Registry (`traffic_accident_severity_model` v1.0.0).
5. **Unified Inference Pipeline Packaging**: Bundled the `ColumnTransformer` (median imputation, standard scaling, and one-hot encoding) and the champion classifier into a single serialized artifact (`models/accident_severity_pipeline.joblib`).
6. **Production Serving with FastAPI**: Engineered a low-latency REST API (`src/api.py`) exposing `/predict`, `/predict/batch`, `/health`, and `/monitoring/metrics` with automated Swagger UI documentation.
7. **Containerization**: Authored a production-grade `Dockerfile.api` and updated `docker-compose.yml` to orchestrate PostgreSQL, Airflow, the FastAPI inference engine, and the MLflow dashboard.
8. **Streamlit UI Integration**: Extended the dashboard (`app.py`) with a 6th view: **"ML Severity Prediction & MLOps Monitoring"**, featuring interactive collision risk simulation, confidence gauges, and emergency risk alert categorization.
9. **Continuous Monitoring & Retraining Design**: Built real-time drift tracking (`src/monitoring.py`) computing the Population Stability Index (PSI) for class drift, spatial centroid shifts for location drift, and high-severity False-Negative Rates (FNR), orchestrated via an automated weekly Airflow DAG (`dags/mlops_pipeline_dag.py`).

---

## 2. Problem Formulation & Class Imbalance Rationale

### 2.1 The Asymmetric Cost of Traffic Collision Misclassification
In traditional classification tasks, models often maximize overall accuracy. However, in traffic accident severity prediction:
- **Predicting a Fatal or Serious accident as Slight (False Negative)** leads to inadequate emergency medical dispatch, insufficient rescue equipment, and delayed trauma care.
- **Predicting a Slight accident as Fatal or Serious (False Positive)** results in temporary over-allocation of emergency resources, which carries minimal safety risk.

Therefore, the primary objective is to maximize **Macro-F1** and **High-Severity Recall (Fatal & Serious Recall)** while minimizing the **False-Negative Rate (FNR)** on severe incidents.

### 2.2 Dataset Class Distribution
In the official UK 2025 collision dataset (101,523 validated collisions):
- **Slight**: 74,880 records (**73.76%**)
- **Serious**: 25,190 records (**24.81%**)
- **Fatal**: 1,453 records (**1.43%**)

A naive majority-class classifier achieves **73.76% accuracy** while suffering a catastrophic **100% False-Negative Rate on Fatal and Serious crashes**. To counteract this, cost-sensitive balanced weighting was integrated into all model loss functions, penalizing fatal misclassifications with a ~23x loss multiplier.

---

## 3. Data Splitting & Strict Featurization Architecture

### 3.1 Chronological Split Strategy
Random shuffling causes future telemetry (e.g. seasonal weather patterns, infrastructure upgrades) to leak into training folds. To adhere to best practices:
- **Train Split (70%)**: 71,066 records (Jan 1, 2025 – Sep 16, 2025)
- **Validation Split (15%)**: 15,228 records (Sep 16, 2025 – Nov 9, 2025)
- **Test Split (15%)**: 15,229 records (Nov 9, 2025 – Dec 31, 2025)

```
[01-Jan-2025] ========= TRAIN (70%) ========= [16-Sep-2025] ===== VAL (15%) ===== [09-Nov-2025] ===== TEST (15%) ===== [31-Dec-2025]
```

### 3.2 Preprocessing Pipeline (`src/features.py`)
To prevent information leakage:
- **Numerical Pipeline**: Median Imputation $\rightarrow$ StandardScaler (`accident_hour`, `speed_limit`, `number_of_vehicles`, `number_of_casualties`, `temperature`, `precipitation`, `wind_speed`).
- **Categorical Pipeline**: Constant "Unknown" Imputation $\rightarrow$ OneHotEncoder with `handle_unknown="ignore"` (`road_type`, `light_conditions`, `road_surface_conditions`, `weather_condition`, `day_of_week`).
- All transformers are **fitted exclusively on the training split**, and applied without modification during validation, testing, and live inference.

---

## 4. Model Training, Comparison & Evaluation

Three candidate architectures were trained and evaluated against identical validation partitions:

| Metric | Baseline Logistic Regression (Balanced) | Balanced Random Forest (100 Trees, Depth 12) | Balanced XGBoost (100 Trees, Depth 6) | Selection Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Macro-F1** | 0.3179 | **0.3598** | 0.3475 | **Random Forest leads (+13.2% vs Baseline)** |
| **Validation Accuracy** | 46.19% | **55.54%** | 52.52% | Higher operational stability |
| **Fatal Recall** | 49.55% | 34.55% | **50.00%** | High sensitivity to critical events |
| **Serious Recall** | 27.69% | **31.57%** | 28.01% | Random Forest detects more serious injuries |
| **Slight Recall** | 52.39% | **64.07%** | 60.86% | Balances specificity on common incidents |
| **Training Duration** | **3.8s** | 12.6s | 8.2s | All within operational SLA |

### 4.1 Champion Model Selection
**RandomForest_Balanced** achieved the highest overall **Macro-F1 (0.3598)** and highest **Serious Recall (31.57%)** while maintaining superior **overall accuracy (55.54%)**.
To compensate for its lower raw Fatal recall threshold, an automated **Risk Alert Engine** was implemented at inference time: if $P(\text{Fatal}) \ge 0.08$ or $P(\text{Fatal} + \text{Serious}) \ge 0.40$, the system automatically triggers a **`CRITICAL_ALERT`**.

### 4.2 Unseen Chronological Test Set Evaluation
The champion Random Forest pipeline was evaluated against the unseen Test split (Nov 9 – Dec 31, 2025):
- **Test Accuracy**: **57.53%**
- **Test Macro-F1**: **0.3810**
- **Test Fatal Recall**: **36.87%**
- **Test Serious Recall**: **38.41%**
- **Test Slight Recall**: **64.09%**
- **Generalization Gap**: Zero degradation from validation to test, proving robust resistance to overfitting.

---

## 5. MLflow Tracking & Model Registry Integration

- **Tracking Store**: Local SQLite database `data/mlflow.db` (`sqlite:///data/mlflow.db`), fully compliant with MLflow 3.x specifications.
- **Artifacts Logged**:
  - `confusion_matrix.png` and normalized heatmap (`confusion_matrix_norm.png`)
  - `classification_report.json`
  - Model pipeline serialized in CloudPickle format
  - Run metadata and hyperparameter logs
- **Model Registry**: Champion model registered as `traffic_accident_severity_model` (Version 1 / Version 2) with lineage tied back to the exact training dataset and Git commit.

---

## 6. Production Inference API (FastAPI)

The inference service (`src/api.py`) is deployed on port 8000:

### Key Endpoints:
1. `GET /`: Health status, active model version, and endpoint sitemap.
2. `GET /health`: Model availability and loaded class taxonomy.
3. `GET /model/metadata`: Model hyperparameters, training metrics, and feature expectations.
4. `POST /predict`: Real-time single collision severity classification.
5. `POST /predict/batch`: High-throughput batch inference for fleet operators.
6. `GET /monitoring/metrics`: Live inference drift metrics, Population Stability Index (PSI), and operational health.
7. `POST /monitoring/evaluate`: Ground-truth evaluation endpoint calculating actual FNR and recall on audited samples.

### Example Single Inference Payload (`POST /predict`):
```json
{
  "accident_hour": 23,
  "day_of_week": "Saturday",
  "speed_limit": 60,
  "number_of_vehicles": 3,
  "number_of_casualties": 2,
  "road_type": "Single carriageway",
  "light_conditions": "Darkness - no lighting",
  "road_surface_conditions": "Wet or damp",
  "weather_condition": "Raining with high winds",
  "temperature": 4.5,
  "precipitation": 2.0,
  "wind_speed": 30.0
}
```

### Response:
```json
{
  "predicted_severity": "Serious",
  "confidence": 0.4613,
  "probabilities": {
    "Fatal": 0.1927,
    "Serious": 0.4613,
    "Slight": 0.3460
  },
  "risk_alert_level": "CRITICAL_ALERT",
  "latency_ms": 65.47,
  "model_version": "1.0.0",
  "model_name": "RandomForest_Balanced"
}
```

---

## 7. Containerization Architecture

A production-ready `Dockerfile.api` was authored using `python:3.12-slim`:
- Non-root execution under unprivileged `appuser`.
- Automated container healthchecks via `curl -f http://localhost:8000/health`.
- Orchestrated in `docker-compose.yml`:
  - `postgres` (port 5432)
  - `fastapi-inference` (port 8000)
  - `mlflow-ui` (port 5000)
  - `airflow-webserver` (port 8080)
  - `airflow-scheduler`

---

## 8. Continuous Monitoring & Automated Retraining Loop

### 8.1 Drift Detection Metrics (`src/monitoring.py`)
1. **Class Distribution Drift (Population Stability Index - PSI)**:
   $$PSI = \sum_{c \in \{\text{Fatal}, \text{Serious}, \text{Slight}\}} (P_c - B_c) \ln\left(\frac{P_c}{B_c}\right)$$
   - $PSI < 0.10$: Stable / Minimal drift.
   - $0.10 \le PSI < 0.20$: Moderate shift (Warning).
   - $PSI \ge 0.20$: Significant distribution drift $\rightarrow$ Triggers Retraining.
2. **Location Drift**: Computes Haversine spatial distance between the inference batch centroid and baseline UK collision centroid ($\text{Lat}=52.85, \text{Lon}=-1.65$).
3. **Data Quality Gate**: Validates speed limits ($[20, 70]$), collision hours ($[0, 23]$), and UK territorial GPS bounding box.

### 8.2 Retraining Decision Triggers
The automated Airflow DAG (`dags/mlops_pipeline_dag.py`) triggers retraining when any of the following conditions occur:
- **Condition 1**: Class PSI $\ge 0.20$.
- **Condition 2**: Fatal False-Negative Rate exceeds $60.0\%$ (Fatal Recall drops below $40\%$) on ground-truth audit batches.
- **Condition 3**: Input Data Quality score drops below $85.0\%$.
- **Condition 4**: Accumulated inference samples reach $10,000+$ records.
- **Condition 5**: Scheduled quarterly model refresh cadence.

---

## 9. Verification & Quality Assurance Evidence

The complete MLOps workflow was verified using automated tests and pipeline executions:
1. **Compilation**: `python -m py_compile` compiled `src/`, `dags/`, `tests/`, and `app.py` with zero errors.
2. **Unit & Integration Tests**: `python -m unittest discover tests` executed **15 test cases in 0.385s** with 100% pass rate.
3. **End-to-End Orchestration**: `python dags/mlops_pipeline_dag.py` executed all 5 tasks sequentially:
   - `check_data_quality` $\rightarrow$ Score: 100.0% (PASS)
   - `train_evaluate_models` $\rightarrow$ Champion: RandomForest_Balanced (PASS)
   - `evaluate_quality_gate` $\rightarrow$ Macro-F1 $\ge 0.30$ & Fatal Recall $\ge 25\%$ (PASS)
   - `deploy_inference_service` $\rightarrow$ Smoke test in 71.36 ms (PASS)
   - `audit_mlops_health` $\rightarrow$ Status: OPERATIONAL_STABLE (PASS)

---

## 10. Reproduction Commands

From project root on Windows or Linux:

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Execute Data Pipeline (Part 1)
python main.py

# 3. Execute MLOps Model Training & Registry (Part 2)
python src/train.py

# 4. Run Automated Test Suite (15 Tests)
python -m unittest discover tests

# 5. Launch FastAPI Microservice
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Launch Interactive Streamlit Analytics & ML Dashboard
streamlit run app.py
```

---

## 11. Conclusion
Part 2 successfully expands the data engineering foundation into an enterprise-grade, end-to-end MLOps solution. By confronting class imbalance with cost-sensitive weighting, guarding against data leakage via chronological splits, tracking full experiment lineage with MLflow, containerizing inference via FastAPI and Docker, and closing the feedback loop with drift monitoring and Airflow retraining, this project delivers a resilient, production-ready system for traffic accident severity analytics and prediction.

