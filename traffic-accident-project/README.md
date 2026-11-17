# Traffic Accident Severity Analytics — Part 1 (Data Engineering Pipeline)

A simple end-to-end data engineering pipeline for UK road traffic accident data:
CSV → raw → staging → cleaned/rejected → weather-enriched → PostgreSQL → analytics mart →
Streamlit dashboard, orchestrated with Apache Airflow.

**This is Part 1 only.** No machine learning / prediction / deployment is included.

---

## ⚠️ About the dataset in this repo

`data/raw/accidents_raw.csv` currently contains **synthetically generated sample data**
(created by `scripts/generate_sample_data.py`) with the same columns and general shape as the
real UK STATS19 collision data, deliberately including some invalid rows so the data-quality
checks have something real to catch.

**Before you submit, replace this with the real dataset:**
1. Go to https://www.gov.uk/government/statistics/road-safety-data
2. Download one year's **Collision** CSV file (e.g. 2023 or the latest final year).
3. Save it as `data/raw/accidents_raw.csv`, matching the column names used below (see
   `docs/data_dictionary.md`), or adjust `scripts/ingest.py` if the real column names differ.
4. Re-run the pipeline (Step 3 below).

---

## 1. Prerequisites (Windows)

- **Python 3.10+** — https://www.python.org/downloads/ (tick "Add Python to PATH" during install)
- **PostgreSQL 14+** — https://www.postgresql.org/download/windows/ (remember the password you set for the `postgres` user)
- **VS Code** — https://code.visualstudio.com/
- **Git** (optional, if you want version control) — https://git-scm.com/download/win
- **WSL2** (only needed for Airflow — see Step 5) — Airflow does not officially support native Windows.

---

## 2. Project setup

Open the project folder in VS Code, then open a terminal (`` Ctrl+` ``, PowerShell by default) and run:

```powershell
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

> Note: `requirements.txt` includes `apache-airflow`. If installing it on native Windows fails
> (it usually will — Airflow needs Linux), that's expected. Skip it here and install Airflow
> only inside WSL2 as described in Step 5. Everything else (pandas, requests, streamlit,
> sqlalchemy, psycopg2) installs fine natively on Windows.

### Set up the PostgreSQL database

Using **pgAdmin** (installed with PostgreSQL) or `psql`, create the database:

```sql
CREATE DATABASE traffic_accidents;
```

If your Postgres username/password/port differ from the defaults (`postgres` / `postgres` / `5432`),
either edit `scripts/config.py` directly, or set environment variables before running scripts:

```powershell
$env:TRAFFIC_DB_USER = "your_username"
$env:TRAFFIC_DB_PASSWORD = "your_password"
```

---

## 3. Run the pipeline manually (recommended first — before adding Airflow)

Run each stage from the `scripts` folder, in order:

```powershell
cd scripts

python generate_sample_data.py   # only needed once, to create demo raw data
python ingest.py                 # raw -> staging
python clean.py                  # staging -> cleaned + rejected
python weather.py                # cleaned -> weather-enriched (calls Open-Meteo)
python load_database.py          # loads everything into PostgreSQL + builds the mart
```

**Expected output of each step:**
- `generate_sample_data.py` → prints row counts; creates `data/raw/accidents_raw.csv`
- `ingest.py` → prints "Ingest complete. N rows written..."; creates `data/staging/accidents_staging.csv`
- `clean.py` → prints cleaned/rejected counts and a rejection-reason breakdown table
- `weather.py` → prints how many location/date groups were queried from Open-Meteo, plus any failures
- `load_database.py` → prints confirmation for each table loaded, and the final "All done!" message

Check the `logs/` folder — each script writes its own `.log` file there.

---

## 4. Run the dashboard

```powershell
streamlit run dashboard/app.py
```

This opens the dashboard in your browser at `http://localhost:8501`, showing:
1. Total accidents / casualties / vehicles (top metrics)
2. Severity distribution
3. Accidents by hour of day
4. Accidents by weekday
5. Accidents by road type (from the analytics mart)
6. Weather-condition analysis (temperature & precipitation by severity)
7. Accident hotspot map
8. Optional vehicle/casualty analysis

---

## 5. Set up Apache Airflow (WSL2)

Since Airflow doesn't run natively on Windows:

1. Install WSL2: open PowerShell **as Administrator** and run `wsl --install`, then restart.
2. Open the **Ubuntu** terminal (installed by WSL2) and navigate to your project
   (Windows drives are mounted under `/mnt/c/...`), e.g.:
   ```bash
   cd /mnt/c/Users/<you>/Documents/traffic-accident-project
   ```
3. Set up Airflow in a Linux virtual environment:
   ```bash
   python3 -m venv airflow_venv
   source airflow_venv/bin/activate
   pip install pandas requests sqlalchemy psycopg2-binary streamlit
   pip install "apache-airflow==2.9.3" --constraint \
     "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.10.txt"

   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create --username admin --password admin \
     --firstname A --lastname B --role Admin --email admin@example.com
   ```
4. Copy (or symlink) `airflow/accident_pipeline.py` into `~/airflow/dags/`:
   ```bash
   cp airflow/accident_pipeline.py ~/airflow/dags/
   ```
5. Start Airflow (in two terminals, both with the venv activated):
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```
6. Open `http://localhost:8080` in your browser, log in (`admin`/`admin`), find
   `traffic_accident_etl_pipeline` in the DAG list, un-pause it, and trigger it manually
   (the play ▶ button).

**Note for your PostgreSQL connection from WSL2**: if Postgres is installed on Windows (not
inside WSL2), set `TRAFFIC_DB_HOST` to your Windows host IP as seen from WSL2 (find it via
`cat /etc/resolv.conf` inside WSL2, or simpler — also install PostgreSQL inside WSL2 for this
project so everything runs in one environment).

---

## 6. Common errors & fixes

| Error | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | Running a script from the wrong folder | `cd scripts` first, then run the script |
| `psycopg2.OperationalError: could not connect to server` | Postgres isn't running, or wrong host/port | Check the PostgreSQL service is started; check `scripts/config.py` values |
| `FATAL: password authentication failed for user "postgres"` | Wrong password in config | Update `TRAFFIC_DB_PASSWORD` or `scripts/config.py` |
| `requests.exceptions.ConnectionError` in `weather.py` | No internet connection during that run | Re-run `weather.py` once you have internet — it will pick up where it left off (cache) |
| Streamlit shows "Could not connect to the database" | `load_database.py` hasn't been run yet, or Postgres isn't running | Run `load_database.py` first, confirm Postgres is running |
| Airflow install fails on native Windows | Airflow doesn't support native Windows | Use WSL2 (Step 5) |

---

## 7. What to screenshot for your report

1. VS Code Explorer showing the full project folder structure
2. Terminal output of each pipeline script running successfully (Step 3)
3. `logs/*.log` file contents (proof of execution with timestamps)
4. pgAdmin (or `psql`) showing the 4 tables and a `SELECT * LIMIT 10` from `accidents` and `accident_analytics`
5. The rejected-records breakdown printed by `clean.py`, and `data/rejected/accidents_rejected.csv` opened in Excel/VS Code
6. The Airflow UI: DAG graph view, and a successful (green) DAG run
7. The Streamlit dashboard — each of the 8 views
8. `docs/architecture_diagram.md` rendered (via mermaid.live or the VS Code extension)

---

## 8. Part 1 Completion Checklist

| # | Requirement | Status | Where it's implemented |
|---|---|---|---|
| 1 | Public traffic accident dataset (UK Road Safety) | ✅ | `data/raw/accidents_raw.csv` (schema matches STATS19; swap in real download — see warning at top) |
| 2 | Python + Pandas for ingestion/transformation | ✅ | `scripts/ingest.py`, `scripts/clean.py` |
| 3 | Open-Meteo API for weather enrichment | ✅ | `scripts/weather.py` |
| 4 | Apache Airflow orchestration | ✅ | `airflow/accident_pipeline.py` |
| 5 | PostgreSQL storage | ✅ | `sql/schema.sql`, `scripts/load_database.py` |
| 6 | Raw / staging / cleaned / analytical layers | ✅ | `data/raw/`, `data/staging/`, `data/cleaned/`, `accident_analytics` table |
| 7 | Data-quality checks | ✅ | `scripts/clean.py` (6 rules — see `docs/validation_rules.md`) |
| 8 | Rejected/invalid records stored separately | ✅ | `data/rejected/accidents_rejected.csv` + `rejected_accidents` table |
| 9 | At least one analytical data mart/table | ✅ | `accident_analytics` table (built in `load_database.py`) |
| 10 | Streamlit dashboard, 5+ views | ✅ | `dashboard/app.py` — 8 views implemented |
| 11 | Architecture diagram | ✅ | `docs/architecture_diagram.md` |
| 12 | Data dictionary + validation rules | ✅ | `docs/data_dictionary.md`, `docs/validation_rules.md` |
| 13 | Screenshots/evidence of execution | ⬜ | You take these — see Section 7 above, once you run each step yourself |
| 14 | README with setup/execution instructions | ✅ | This file |

**Action needed from you before final submission:** replace the sample dataset with the real
STATS19 download, re-run the full pipeline, and capture the screenshots listed in Section 7.
