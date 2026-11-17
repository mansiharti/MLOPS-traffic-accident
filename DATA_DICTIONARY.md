# Data Dictionary and Validation Rules

Comprehensive data dictionary covering the Kimball Star Schema Warehouse, Analytical Mart, Feature Engineering Pipeline, Model Registry, and MLOps Monitoring metrics.

---

## 1. Kimball Star Schema Dimensional Tables

### 1.1 `dim_date` (Calendar Dimension)
| Column Name | Data Type | Nullable | Description / Business Meaning | Sample / Allowed Range |
| :--- | :--- | :---: | :--- | :--- |
| `date_id` | INTEGER | No (PK) | Surrogate primary key | Auto-incrementing integer |
| `accident_date` | TEXT | No (Unique)| Collision date in ISO-8601 | `2025-01-01` to `2025-12-31` |
| `day_of_week` | TEXT | Yes | Full day name | `Monday` to `Sunday` |
| `day` | INTEGER | Yes | Calendar day of month | `1` to `31` |
| `month` | INTEGER | Yes | Calendar month number | `1` to `12` |
| `year` | INTEGER | Yes | Calendar year | `2025` |
| `is_weekend` | INTEGER | Yes | Weekend indicator flag | `0` (Weekday), `1` (Weekend) |

### 1.2 `dim_location` (Geospatial Dimension)
| Column Name | Data Type | Nullable | Description / Business Meaning | Sample / Allowed Range |
| :--- | :--- | :---: | :--- | :--- |
| `location_id` | INTEGER | No (PK) | Surrogate primary key | Auto-incrementing integer |
| `local_authority`| TEXT | No | Municipal district code or administrative name | `E06000001`, `Leeds`, `Birmingham` |
| `latitude` | REAL | No | WGS84 geographic latitude coordinate | $49.0 \le \text{Lat} \le 61.5$ (UK bounds) |
| `longitude` | REAL | No | WGS84 geographic longitude coordinate | $-8.5 \le \text{Lon} \le 2.5$ (UK bounds) |

### 1.3 `dim_road` (Roadway Environment Dimension)
| Column Name | Data Type | Nullable | Description / Business Meaning | Sample / Allowed Range |
| :--- | :--- | :---: | :--- | :--- |
| `road_id` | INTEGER | No (PK) | Surrogate primary key | Auto-incrementing integer |
| `road_type` | TEXT | Yes | Roadway configuration | `Single carriageway`, `Dual carriageway`, `Roundabout`, `One way street`, `Slip road` |
| `speed_limit` | INTEGER | Yes | Posted regulatory speed limit in mph | `20`, `30`, `40`, `50`, `60`, `70` |
| `light_conditions`| TEXT | Yes | Natural and artificial lighting | `Daylight`, `Darkness - lights lit`, `Darkness - no lighting`, `Darkness - lights unlit` |
| `road_surface_conditions`| TEXT | Yes | Pavement moisture and traction condition | `Dry`, `Wet or damp`, `Snow`, `Frost or ice`, `Flood over 3cm deep` |

### 1.4 `dim_weather` (Meteorological Dimension)
| Column Name | Data Type | Nullable | Description / Business Meaning | Sample / Allowed Range |
| :--- | :--- | :---: | :--- | :--- |
| `weather_id` | INTEGER | No (PK) | Surrogate primary key | Auto-incrementing integer |
| `weather_code` | INTEGER | Yes | WMO standardized weather classification code | `0` to `99` |
| `weather_condition`| TEXT | Yes | Human-readable atmospheric description | `Fine no high winds`, `Raining no high winds`, `Snowing no high winds`, `Fog or mist` |
| `temperature` | REAL | Yes | 2m ambient air temperature (°C) | Continuous real value (e.g. `10.0`) |
| `precipitation`| REAL | Yes | Total liquid equivalent precipitation (mm) | $\ge 0.0$ |
| `rain` | REAL | Yes | Liquid rain accumulation (mm) | $\ge 0.0$ |
| `snowfall` | REAL | Yes | Solid snowfall accumulation (cm) | $\ge 0.0$ |
| `wind_speed` | REAL | Yes | 10m surface wind velocity (km/h) | $\ge 0.0$ |

### 1.5 `dim_severity` (Controlled Classification Dimension)
| Column Name | Data Type | Nullable | Description / Business Meaning | Allowed Values |
| :--- | :--- | :---: | :--- | :--- |
| `severity_id` | INTEGER | No (PK) | Surrogate primary key | `1`, `2`, `3` |
| `severity_name` | TEXT | No (Unique)| Official UK DfT injury severity taxonomy | `Fatal`, `Serious`, `Slight` |

### 1.6 `fact_accident` (Fact Table)
| Column Name | Data Type | Constraints | Description / Grain |
| :--- | :--- | :--- | :--- |
| `accident_index` | TEXT | PRIMARY KEY | Unique collision incident identifier |
| `date_id` | INTEGER | FK $\rightarrow$ `dim_date.date_id` | Foreign key referencing calendar date |
| `location_id` | INTEGER | FK $\rightarrow$ `dim_location.location_id` | Foreign key referencing incident geography |
| `road_id` | INTEGER | FK $\rightarrow$ `dim_road.road_id` | Foreign key referencing road environment |
| `weather_id` | INTEGER | FK $\rightarrow$ `dim_weather.weather_id` | Foreign key referencing weather telemetry |
| `severity_id` | INTEGER | FK $\rightarrow$ `dim_severity.severity_id` | Foreign key referencing collision severity |
| `accident_time` | TEXT | Not Null | Degenerate dimension: collision time (`HH:MM`) |
| `accident_hour` | INTEGER | Not Null | Derived collision hour ($0-23$) |
| `number_of_vehicles`| INTEGER | Not Null | Metric: count of involved vehicles ($\ge 1$) |
| `number_of_casualties`| INTEGER| Not Null | Metric: count of injured casualties ($\ge 1$) |

---

## 2. Serving Data Mart (`accident_analytics`)
Flattened, denormalized table used for sub-second dashboard rendering and model training feature extraction:
- `accident_index`, `accident_date`, `accident_time`, `accident_hour`, `day_of_week`
- `accident_severity` (`Fatal`, `Serious`, `Slight`)
- `number_of_vehicles`, `number_of_casualties`
- `road_type`, `speed_limit`, `light_conditions`, `road_surface_conditions`
- `local_authority`, `latitude`, `longitude`
- `weather_code`, `weather_condition`, `temperature`, `precipitation`, `rain`, `snowfall`, `wind_speed`

---

## 3. Part 2 Feature Engineering & MLOps Schema

### 3.1 Model Features Specification (`ALL_FEATURES`)
| Feature Name | Feature Group | Modeling Treatment | Imputation Strategy | Encoding / Scaling |
| :--- | :--- | :--- | :--- | :--- |
| `accident_hour` | Numerical | Temporal density | SimpleImputer(median) | StandardScaler |
| `speed_limit` | Numerical | Kinetic hazard indicator | SimpleImputer(median) | StandardScaler |
| `number_of_vehicles`| Numerical | Crash complexity | SimpleImputer(median) | StandardScaler |
| `number_of_casualties`| Numerical | Human exposure metric | SimpleImputer(median) | StandardScaler |
| `temperature` | Numerical | Thermal environment | SimpleImputer(median) | StandardScaler |
| `precipitation` | Numerical | Moisture accumulation | SimpleImputer(median) | StandardScaler |
| `wind_speed` | Numerical | Aerodynamic lateral force | SimpleImputer(median) | StandardScaler |
| `road_type` | Categorical | Infrastructure classification | SimpleImputer("Unknown") | OneHotEncoder(ignore) |
| `light_conditions` | Categorical | Visibility context | SimpleImputer("Unknown") | OneHotEncoder(ignore) |
| `road_surface_conditions`| Categorical | Pavement friction factor | SimpleImputer("Unknown") | OneHotEncoder(ignore) |
| `weather_condition`| Categorical | Environmental conditions | SimpleImputer("Unknown") | OneHotEncoder(ignore) |
| `day_of_week` | Categorical | Behavioral weekly rhythm | SimpleImputer("Unknown") | OneHotEncoder(ignore) |

### 3.2 Target Variable (`TARGET_COLUMN`)
- **Target**: `accident_severity`
- **Classes**:
  - `Fatal` (Class 0 / Label 'Fatal'): Death within 30 days of incident.
  - `Serious` (Class 1 / Label 'Serious'): Severe injury requiring hospitalization (fractures, severe lacerations).
  - `Slight` (Class 2 / Label 'Slight'): Minor injury (bruises, sprains, shock) not requiring overnight admission.

---

## 4. MLOps Drift & Operational Monitoring Metrics

| Metric | Target Dimension | Computation Formula / Rule | Threshold & Action |
| :--- | :--- | :--- | :--- |
| **Population Stability Index (PSI)** | Class Distribution Drift | $PSI = \sum_{c} (P_c - B_c) \ln(P_c / B_c)$ comparing inference predictions against baseline training distribution | - $PSI < 0.10$: Stable<br/>- $0.10 \le PSI < 0.20$: Warning<br/>- $PSI \ge 0.20$: **Trigger Automated Retraining** |
| **Spatial Centroid Drift** | Location Drift | Haversine distance in km between current inference batch spatial centroid and baseline UK centroid ($\text{Lat}=52.85, \text{Lon}=-1.65$) | - $< 50\text{ km}$: Stable<br/>- $\ge 120\text{ km}$: Regional Shift Alert |
| **Fatal False-Negative Rate (FNR)** | Model Performance Safety Gate | $FNR = 1.0 - \text{Recall}_{\text{Fatal}}$ on audited ground truth batches | - $FNR \ge 60.0\%$ (Fatal Recall $< 40\%$): **Trigger Safety Retraining** |
| **Data Quality Score** | Data Pipeline Integrity | $100 \times (1.0 - \frac{\text{Violations}}{\text{Records} \times \text{Rules}})$ auditing boundaries, missing fields, and coordinates | - $< 85.0\%$: **Data Pipeline Ingestion Halt Alert** |
| **Inference Latency** | Service SLA | Wall-clock execution time in milliseconds | - Target: $\le 100\text{ ms}$ per request |

---

## 5. Ingestion Validation and Quarantine Rules

The quality gate in `scripts/pipeline.py` quarantines non-compliant records to `data/output/rejected_records.csv` with explicit failure codes:
1. `duplicate_record`: Repeated `accident_index` after the initial occurrence.
2. `missing_accident_id`: Blank, whitespace, or null identifier.
3. `missing_date_or_time` or `invalid_date_format`: Absent or unparseable temporal attributes.
4. `missing_or_invalid_severity`: Value outside `Fatal`, `Serious`, or `Slight`.
5. `zero_coordinates` or `out_of_bounds_coordinates`: Lat/Lon outside UK territory or $(0.0, 0.0)$.
6. `missing_road_type`, `missing_number_of_vehicles`, or `missing_number_of_casualties`: Missing required collision attributes.
