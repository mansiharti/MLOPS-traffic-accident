# Traffic Accident Severity Analytics and Prediction
## Course: Data Engineering and MLOps | Complete Implementation (Parts 1 & 2)

An enterprise-grade, end-to-end Data Engineering pipeline, Kimball Star Schema dimensional warehouse, interactive Streamlit analytics application, and production MLOps system for UK road traffic collision severity analytics, classification, and continuous drift monitoring.

---

## 1. Project Submission Package Checklist

| Item | Required Deliverable | Description & File Locations | Part |
| :---: | :--- | :--- | :---: |
| **1** | **Source Code & Scripts** | - Data Engineering: [scripts/pipeline.py](scripts/pipeline.py), [scripts/weather_enrichment.py](scripts/weather_enrichment.py), [scripts/load_db.py](scripts/load_db.py), [main.py](main.py)<br/>- MLOps Core: [src/features.py](src/features.py), [src/train.py](src/train.py), [src/predict.py](src/predict.py), [src/monitoring.py](src/monitoring.py), [src/api.py](src/api.py) | **1 & 2** |
| **2** | **Airflow DAGs** | - Part 1 Data Pipeline DAG: [dags/traffic_accident_dag.py](dags/traffic_accident_dag.py)<br/>- Part 2 MLOps Retraining DAG: [dags/mlops_pipeline_dag.py](dags/mlops_pipeline_dag.py) | **1 & 2** |
| **3** | **Database Schema & Populated Tables** | - PostgreSQL DDL: [schema.sql](schema.sql)<br/>- SQLite Populated Warehouse (101,523 rows): [data/traffic_accidents.db](data/traffic_accidents.db) | **1** |
| **4** | **Dataset Info & Access Protocols** | [DATA_SOURCES.md](DATA_SOURCES.md) | **1** |
| **5** | **Architecture & Data Flow** | [ARCHITECTURE.md](ARCHITECTURE.md) (Interactive Mermaid diagrams for Data Engineering & MLOps layers) | **1 & 2** |
| **6** | **Data Dictionary & Validation Rules** | [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | **1** |
| **7** | **Interactive Streamlit Dashboard** | [app.py](app.py) (6 views including Real-Time ML Risk Prediction, Confusion Matrix & Drift Gauges) | **1 & 2** |
| **8** | **Project Reports (8-12 pages)** | - Master Combined Report: [REPORT.md](REPORT.md)<br/>- Dedicated Part 2 MLOps Report: [docs/PART2_MLOPS_REPORT.md](docs/PART2_MLOPS_REPORT.md) | **1 & 2** |
| **9** | **Execution Evidence & Audit Logs** | [EXECUTION_EVIDENCE.md](EXECUTION_EVIDENCE.md) | **1 & 2** |
| **10**| **FastAPI REST Microservice & Docker** | - REST API: [src/api.py](src/api.py) (Port 8000, Swagger UI at `/docs`)<br/>- Containers: [Dockerfile.api](Dockerfile.api), [docker-compose.yml](docker-compose.yml) | **2** |
| **11**| **Automated Test Suite** | [tests/test_pipeline.py](tests/test_pipeline.py), [tests/test_mlops.py](tests/test_mlops.py) (15 unit & integration tests) | **1 & 2** |

---

## 2. Quickstart & Execution Guide

### Prerequisites
- Python 3.11 or 3.12
- PowerShell / Bash
- (Optional) Docker & Docker Compose for containerized execution

### Step 1: Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 2: Part 1 - Data Engineering Pipeline
Ingests raw UK 2025 collisions (`data/raw/uk_2025/`), joins casualty and vehicle detail records, enforces boundary validation, applies meteorological enrichment, and loads the Star Schema warehouse:
```powershell
python main.py
```
To run the ordered Airflow tasks sequentially:
```powershell
python dags\traffic_accident_dag.py
```

---

### Step 3: Part 2 - MLOps Model Training, Comparison & Registry
Applies chronological train/val/test splitting (70%/15%/15%), trains Baseline Logistic Regression, Balanced Random Forest, and Balanced XGBoost, logs all runs to MLflow (`data/mlflow.db`), compares candidates, evaluates champion on unseen test data, and registers `traffic_accident_severity_model`:
```powershell
python src\train.py
```
To run the automated weekly MLOps Airflow retraining DAG:
```powershell
python dags\mlops_pipeline_dag.py
```

---

### Step 4: Run Automated Unit & Integration Tests
Executes 15 comprehensive tests covering data cleaning, chronological split validation, model loading, single & batch inference, data quality scoring, PSI class drift computation, location centroid drift, retraining triggers, and FastAPI endpoints:
```powershell
python -m unittest discover tests
```

---

### Step 5: Launch FastAPI Production REST API
Starts the high-performance model serving microservice on port 8000:
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at:
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Healthcheck**: [http://localhost:8000/health](http://localhost:8000/health)
- **Monitoring Metrics**: [http://localhost:8000/monitoring/metrics](http://localhost:8000/monitoring/metrics)

#### Example cURL Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

---

### Step 6: Launch Streamlit Analytics & ML Prediction Dashboard
Starts the unified Streamlit application:
```powershell
streamlit run app.py
```
Open your browser at: **http://localhost:8501**

#### Dashboard Views:
1. **Severity & Temporal Patterns**: Donut severity breakdown, hourly peaks, monthly trends, and day-of-week incident frequency.
2. **Location Hotspot Map**: Interactive density map with UK municipal authority rankings.
3. **Weather & Road Conditions**: Incident distribution filtered by surface moisture, lighting, temperature, and wind speed.
4. **Vehicle & Casualty Analysis**: Casualties vs posted speed limits and data-quality quarantine audit breakdown.
5. **Data Mart Explorer**: Filterable, searchable record explorer with one-click CSV export.
6. **ML Severity Prediction & MLOps Monitoring**:
   - **Live Risk Inference**: Interactive scenario simulation with real-time class probability distribution and emergency risk alert categorization (`CRITICAL_ALERT`, `ELEVATED_RISK`, `STANDARD_RISK`).
   - **MLflow Model Registry**: Model hyperparameters, candidate benchmark table, and test confusion matrix display.
   - **Data Drift & Health Monitoring**: Live Population Stability Index (PSI) gauge, spatial centroid drift tracking, and retraining decision checklist.

---

### Step 7: Launch MLflow Tracking UI
To inspect experiment runs, parameters, metrics, and registered model versions:
```powershell
mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000
```
Open your browser at: **http://localhost:5000**

---

### Step 8: Multi-Container Unified Docker Deployment (Parts 1 & 2)
To launch the entire combined platform (PostgreSQL Star Schema, Automatic Warehouse Loader, FastAPI Prediction API, Streamlit BI & ML Dashboard, MLflow Tracking UI, and Airflow Pipeline Scheduler) with a single command:
```bash
docker compose up --build
```

#### Services Available Once Running:
| Service | Component | Local URL / Access | Description |
| :--- | :--- | :--- | :--- |
| **Streamlit Dashboard** | **Parts 1 & 2** | [http://localhost:8501](http://localhost:8501) | Full interactive BI (Views 1–5) + Real-Time ML Risk Simulator (View 6) |
| **FastAPI REST Service** | **Part 2** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI for real-time `/predict`, `/health`, and drift metrics |
| **MLflow Tracking UI** | **Part 2** | [http://localhost:5000](http://localhost:5000) | Candidate model tournament, parameters, metrics, confusion matrices & registry |
| **Airflow Webserver** | **Parts 1 & 2** | [http://localhost:8080](http://localhost:8080) | Orchestrates Part 1 ETL DAG & Part 2 MLOps Retraining DAG (User: `admin` / `admin`) |
| **PostgreSQL Warehouse** | **Part 1** | `localhost:5432` | Kimball Star Schema (`fact_accident`) & denormalized `accident_analytics` mart |

---

## 3. Directory Layout

```
traffic-accident-project/
├── .streamlit/
│   └── config.toml                # Clean UI configuration
├── dags/
│   ├── traffic_accident_dag.py    # Part 1: Data ingestion & Star Schema ETL DAG
│   └── mlops_pipeline_dag.py      # Part 2: MLOps data quality, training & deploy DAG
├── data/
│   ├── raw/
│   │   ├── uk_2025/               # Official DfT collision, vehicle, and casualty CSVs
│   │   └── accidents_raw.csv      # Immutable raw landing extract
│   ├── output/
│   │   ├── accidents_cleaned.csv  # Validated records (101,523 rows)
│   │   ├── accidents_enriched.csv # Joined meteorological telemetry
│   │   ├── rejected_records.csv   # Quarantine audit log with rejection reasons
│   │   └── weather_cache.json     # Request cache for Open-Meteo API
│   ├── traffic_accidents.db       # Embedded SQLite Star Schema Warehouse & Data Mart
│   └── mlflow.db                  # MLflow 3.x tracking and Model Registry backend store
├── docs/
│   ├── PART2_MLOPS_REPORT.md      # Standalone Part 2 MLOps Technical Report
│   ├── architecture_diagram.md
│   └── data_dictionary.md
├── models/
│   ├── accident_severity_pipeline.joblib  # Serialized Champion Preprocessing + Model Pipeline
│   ├── baseline_distribution.json         # Reference distributions for PSI drift detection
│   ├── model_metadata.json                # Version, hyperparameters, validation & test metrics
│   └── artifacts/                         # Confusion matrices, normalized heatmaps, reports
├── scripts/
│   ├── pipeline.py                # Raw ingestion, detail join, and validation gate
│   ├── weather_enrichment.py      # Open-Meteo historical weather integration
│   └── load_db.py                 # Dimensional Star Schema & Mart loader (Postgres/SQLite)
├── src/
│   ├── __init__.py
│   ├── features.py                # Chronological split & scikit-learn ColumnTransformer
│   ├── train.py                   # Multi-model training, MLflow tracking, registry & export
│   ├── predict.py                 # Low-latency inference engine with risk alert scoring
│   ├── monitoring.py              # Real-time PSI class drift, spatial drift, and FNR auditor
│   └── api.py                     # Production FastAPI REST microservice
├── tests/
│   ├── test_pipeline.py           # Part 1 ETL & database unit tests
│   └── test_mlops.py              # Part 2 MLOps, features, models, drift & API tests
├── app.py                         # Unified 6-view Streamlit interactive dashboard
├── main.py                        # Part 1 master end-to-end pipeline runner
├── schema.sql                     # PostgreSQL DDL for Star Schema & accident_analytics mart
├── requirements.txt               # Unified project dependencies
├── Dockerfile.api                 # Production container for FastAPI inference service
├── Dockerfile                     # Apache Airflow container definition
├── docker-compose.yml             # Multi-service container orchestration
├── ARCHITECTURE.md                # System Architecture & Mermaid Diagrams
├── DATA_DICTIONARY.md             # Complete data dictionary & validation rules
├── DATA_SOURCES.md                # Public data sources & access protocols
├── EXECUTION_EVIDENCE.md          # Verified execution logs & table counts
└── REPORT.md                      # Comprehensive Master Project Report (Parts 1 & 2)
```

---

## 4. Key Performance Highlights

- **ETL Scale**: 101,525 raw UK collisions processed $\rightarrow$ 101,523 clean records in warehouse.
- **Data Quality**: 99.998% pass rate; invalid records quarantined with audit reasons.
- **Model Benchmark**: Balanced Random Forest outperformed baseline with **0.3598 Val Macro-F1**, **0.3810 Test Macro-F1**, and **57.53% Test Accuracy**.
- **Safety Alerting**: Emergency risk thresholding triggers alerts when $P(\text{Fatal}) \ge 0.08$ or $P(\text{Fatal} + \text{Serious}) \ge 0.40$, successfully combating class imbalance false negatives.
- **Inference Latency**: Real-time REST single prediction served in **~65 ms**.
- **Test Suite**: 15 unit and integration tests passing in **0.385s**.
