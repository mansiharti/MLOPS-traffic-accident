# Data Dictionary — Traffic Accident Severity Analytics

## 1. `accidents` table (cleaned fact table)

| Column | Type | Source | Description |
|---|---|---|---|
| accident_index | VARCHAR(50), PK | STATS19 raw file | Unique ID for the accident record |
| accident_date | DATE | STATS19 raw file | Date the accident occurred |
| accident_time | VARCHAR(5) | STATS19 raw file | Time of the accident, 24hr `HH:MM` |
| day_of_week | VARCHAR(15) | STATS19 raw file | Day name (Monday–Sunday) |
| accident_severity | VARCHAR(10) | STATS19 raw file | One of `Fatal`, `Serious`, `Slight` |
| number_of_vehicles | INTEGER | STATS19 raw file | Count of vehicles involved |
| number_of_casualties | INTEGER | STATS19 raw file | Count of casualties recorded |
| road_type | VARCHAR(50) | STATS19 raw file | e.g. Single carriageway, Roundabout |
| speed_limit | INTEGER | STATS19 raw file | Posted speed limit (mph) at the location |
| light_conditions | VARCHAR(50) | STATS19 raw file | Daylight / darkness variants |
| road_surface_conditions | VARCHAR(50) | STATS19 raw file | Dry / Wet or damp / Snow / Frost or ice |
| latitude | DOUBLE PRECISION | STATS19 raw file | WGS84 latitude of the accident |
| longitude | DOUBLE PRECISION | STATS19 raw file | WGS84 longitude of the accident |
| local_authority | VARCHAR(100) | STATS19 raw file | Local authority / district name |
| temperature_c | DOUBLE PRECISION | Open-Meteo API | Daily mean temperature (°C) at accident's rounded location/date |
| precipitation_mm | DOUBLE PRECISION | Open-Meteo API | Daily total precipitation (mm) |
| windspeed_max_kmh | DOUBLE PRECISION | Open-Meteo API | Daily max wind speed (km/h) |
| loaded_at | TIMESTAMP | Pipeline-generated | When the row was loaded into Postgres |

## 2. `weather` table

| Column | Type | Description |
|---|---|---|
| weather_id | SERIAL, PK | Internal ID |
| rounded_lat / rounded_lon | DOUBLE PRECISION | Location rounded to 1 decimal place (~11km) — the "representative location" used to group nearby accidents for one weather lookup |
| weather_date | DATE | Date the weather values apply to |
| temperature_c, precipitation_mm, windspeed_max_kmh | DOUBLE PRECISION | Same meaning as in `accidents`, sourced from Open-Meteo |

## 3. `rejected_accidents` table

| Column | Type | Description |
|---|---|---|
| id | SERIAL, PK | Internal ID |
| accident_index, accident_date, accident_time, accident_severity, road_type, latitude, longitude | Various | Copied as-is (as text) from the raw/staging data — no type conversion, since the row may be invalid |
| rejection_reasons | TEXT | Comma-separated list of every rule this row failed (see `validation_rules.md`) |
| loaded_at | TIMESTAMP | When the row was loaded |

## 4. `accident_analytics` table (analytical data mart)

One row per unique `(accident_date, road_type, accident_severity)` combination.

| Column | Type | Description |
|---|---|---|
| id | SERIAL, PK | Internal ID |
| accident_date | DATE | Grouping key |
| road_type | VARCHAR(50) | Grouping key |
| accident_severity | VARCHAR(10) | Grouping key |
| total_accidents | INTEGER | COUNT of accidents in this group |
| total_vehicles_involved | INTEGER | SUM of vehicles involved |
| total_casualties | INTEGER | SUM of casualties |
| avg_temperature_c | DOUBLE PRECISION | AVG temperature across the group |
| avg_precipitation_mm | DOUBLE PRECISION | AVG precipitation across the group |
| generated_at | TIMESTAMP | When the mart was last rebuilt |

## Notes on data provenance

The sample data shipped in `data/raw/accidents_raw.csv` is **synthetically generated** (see
`scripts/generate_sample_data.py`) to have the same shape/columns as the real UK STATS19
collision dataset, because this build environment could not reach gov.uk to download it directly.
**Before final submission, replace this file with the real STATS19 download** from
https://www.gov.uk/government/statistics/road-safety-data — the rest of the pipeline requires
no code changes as long as column names match (adjust `ingest.py`'s column handling if the
real file uses different column names, e.g. `Accident_Index` instead of `accident_index`).
