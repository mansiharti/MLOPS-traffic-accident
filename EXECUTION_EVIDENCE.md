# Project Execution Evidence & Verification Log

Verified across both Part 1 (Data Engineering Warehouse) and Part 2 (MLOps Pipeline Extension).

---

## 1. Part 1 Data Engineering Pipeline Execution

### Command:
```powershell
python main.py
python dags\traffic_accident_dag.py
```

### Observed Execution Log:
```
=================================================================
[*] TRAFFIC ACCIDENT SEVERITY & WEATHER ANALYTICS PIPELINE
=================================================================

[STEP 1/3] Ingesting & Cleaning Raw Accidents...
-> Found UK 2025 collision dataset: data\raw\uk_2025\collisions.csv
-> Joined vehicles.csv and casualties.csv to collision records
==================================================
TRAFFIC ACCIDENT PIPELINE: UK 2025 CLEANING SUMMARY
==================================================
Rows before cleaning: 101,525
Rows after cleaning:  101,523
Duplicates removed:   0
Invalid records:      2
Cleaned output: data\output\accidents_cleaned.csv

[STEP 2/3] Enriching with Historical Weather from Open-Meteo...
[Weather Enrichment] Used recorded collision-scene weather for 101,523 UK 2025 records.
[Weather Enrichment] Output saved to: data/output/accidents_enriched.csv

[STEP 3/3] Loading Dimensional Star Schema & Analytical Mart...
[Database Loader] Processing 101523 records from data/output/accidents_enriched.csv...
[Database Loader] Populating SQLite tables...
  -> Verification Table Counts:
     - dim_date            : 365 records
     - dim_location        : 101388 records
     - dim_road            : 517 records
     - dim_weather         : 9 records
     - dim_severity        : 3 records
     - fact_accident       : 101523 records
     - accident_analytics  : 101523 records

[Database Loader] Finished successfully! Analytical warehouse ready.

=================================================================
[+] PIPELINE EXECUTION SUMMARY & HEALTH REPORT
=================================================================
Total Collisions Processed : 101,523
Total Casualties Recorded  : 127,881
Severity Distribution      : Fatal: 1453 | Serious: 25190 | Slight: 74880
Analytical Database        : data\traffic_accidents.db
Execution Completed in     : 63.69 seconds
=================================================================
```

---

## 2. Part 2 MLOps Model Training & Registry Execution

### Command:
```powershell
python src\train.py
```

### Observed Execution Log:
```
=================================================================
[*] TRAFFIC ACCIDENT SEVERITY: END-TO-END MLOPS TRAINING PIPELINE
=================================================================
[Data Loader] Loaded 101,523 records from SQLite analytical warehouse (data\traffic_accidents.db).
[Chronological Split] Total: 101,523 rows
  - Train : 71,066 rows (2025-01-01 to 2025-09-16)
  - Val   : 15,228 rows (2025-09-16 to 2025-11-09)
  - Test  : 15,229 rows (2025-11-09 to 2025-12-31)
[*] Exported models/baseline_distribution.json for continuous drift monitoring.

---> Training & Evaluating Candidate: Baseline_LogisticRegression...
    Validation Macro-F1   : 0.3179
    Validation Accuracy   : 0.4619
    Fatal Recall (Alert)  : 49.55% (FNR: 50.45%)
    Serious Recall        : 27.69%
    Slight Recall         : 52.39%

---> Training & Evaluating Candidate: RandomForest_Balanced...
    Validation Macro-F1   : 0.3598
    Validation Accuracy   : 0.5554
    Fatal Recall (Alert)  : 34.55% (FNR: 65.45%)
    Serious Recall        : 31.57%
    Slight Recall         : 64.07%

---> Training & Evaluating Candidate: XGBoost_Balanced...
    Validation Macro-F1   : 0.3475
    Validation Accuracy   : 0.5252
    Fatal Recall (Alert)  : 50.00% (FNR: 50.00%)
    Serious Recall        : 28.01%
    Slight Recall         : 60.86%

=================================================================
[+] MODEL COMPARISON TABLE (Validation Performance):
=================================================================
                            val_macro_f1 val_fatal_recall val_serious_recall val_accuracy fatal_fnr
RandomForest_Balanced           0.359755         0.345455           0.315679     0.555424  0.654545
XGBoost_Balanced                0.347487              0.5           0.280105     0.525151       0.5
Baseline_LogisticRegression     0.317908         0.495455           0.276943     0.461912  0.504545

[CHAMPION] SELECTED CHAMPION MODEL: RandomForest_Balanced (Run ID: 1aee3645527c447e90a7049fcfed27fd)

---> Evaluating Champion Model on Unseen Chronological TEST Set...
  Test Accuracy         : 0.5753
  Test Macro-F1         : 0.3810
  Test Fatal Recall     : 36.87%
  Test Serious Recall   : 38.41%
  Test Slight Recall    : 64.09%
[+] Model registered in MLflow Registry as 'traffic_accident_severity_model' (Version: 2)
[+] Exported Champion Pipeline Artifact to: models\accident_severity_pipeline.joblib
[+] Exported Model Metadata Manifest to: models/model_metadata.json

=================================================================
[SUCCESS] MLOPS MODEL TRAINING & REGISTRY COMPLETE!
=================================================================
```

---

## 3. Airflow MLOps Orchestration Pipeline Execution

### Command:
```powershell
python dags\mlops_pipeline_dag.py
```

### Observed Execution Log:
```
=================================================================
[*] EXECUTING MLOPS RETRAINING & DEPLOYMENT PIPELINE TASKS
=================================================================
[MLOps Task 1/5] Auditing Warehouse Data Quality & Feature Drift...
Data Quality Audit Result: Score = 100.0% (Status: PASS)

[MLOps Task 2/5] Executing Chronological Split, Balanced Model Training & MLflow Tracking...
Champion Model Selected: RandomForest_Balanced (Val Macro-F1: 0.3598)

[MLOps Task 3/5] Evaluating Champion Model against Production Quality Gates...
Quality Gate Checks:
  - Test Macro-F1       : 0.3810 (Required >= 0.30) -> PASS
  - Test Fatal Recall   : 36.87% (Required >= 25.0%) -> PASS
Quality Gates Passed! Model approved for production deployment.

[MLOps Task 4/5] Deploying and validating active inference pipeline...
Inference smoke test successful: Predicted 'Slight' in 71.36 ms.

[MLOps Task 5/5] Finalizing MLOps Lifecycle Audit & Health Status...
MLOps Status: OPERATIONAL_STABLE

=================================================================
[SUCCESS] ALL MLOPS PIPELINE TASKS EXECUTED SUCCESSFULLY!
=================================================================
```

---

## 4. Automated Unit & Integration Test Suite Execution

### Command:
```powershell
python -m unittest discover tests
```

### Observed Output:
```
...............
----------------------------------------------------------------------
Ran 15 tests in 0.385s

OK
```

All 15 tests passed with zero failures across data quality validation, chronological splits, model inference, drift assessment, retraining decision triggers, and FastAPI REST endpoints.
