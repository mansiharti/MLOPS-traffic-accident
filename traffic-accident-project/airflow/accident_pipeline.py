"""
accident_pipeline.py
---------------------
Airflow DAG that orchestrates the full Traffic Accident ETL pipeline.

WHY AIRFLOW HERE:
Airflow lets us define the pipeline as a sequence of steps ("tasks") with
clear dependencies (task B only runs after task A succeeds), and gives us
a visual UI to monitor runs, see logs, and re-run failed steps - which is
exactly what the assignment asks for ("Uses Apache Airflow to orchestrate
the ETL pipeline").

DESIGN CHOICE - KEEPING IT SIMPLE:
Each Airflow task simply calls the `main()` function that already exists
in our plain scripts/*.py files (ingest.py, clean.py, weather.py,
load_database.py). We are NOT rewriting our logic inside Airflow - Airflow
is only responsible for calling our existing, already-tested scripts in
the right order. This keeps the "real" logic simple, readable, and
testable on its own (you can run each script directly with `python
script.py`, completely without Airflow, exactly as you did in earlier
steps), while Airflow handles scheduling/orchestration/monitoring on top.

IMPORTANT - WINDOWS NOTE:
Apache Airflow does not officially support running natively on Windows.
You need to run it inside WSL2 (Windows Subsystem for Linux) or Docker.
See README.md for exact setup instructions. The DAG code itself below
does not change based on how you run Airflow.
"""

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make sure Python can find our existing scripts/ folder and import from it,
# no matter where Airflow itself is installed.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import ingest
import clean
import weather
import load_database


default_args = {
    "owner": "student",
    "retries": 1,
}

with DAG(
    dag_id="traffic_accident_etl_pipeline",
    description="Traffic Accident Severity Analytics - Part 1 ETL pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # run manually from the Airflow UI for this project
    catchup=False,
    tags=["data-engineering", "mlops-course"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_raw_to_staging",
        python_callable=ingest.main,
    )

    clean_task = PythonOperator(
        task_id="clean_staging_to_cleaned_rejected",
        python_callable=clean.main,
    )

    weather_task = PythonOperator(
        task_id="enrich_with_weather",
        python_callable=weather.main,
    )

    load_task = PythonOperator(
        task_id="load_to_postgres_and_build_mart",
        python_callable=load_database.main,
    )

    # Define the order: this is the ETL flow from the assignment brief.
    ingest_task >> clean_task >> weather_task >> load_task
