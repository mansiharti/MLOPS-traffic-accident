-- ============================================================
-- schema.sql
-- ============================================================
-- Simple schema for the Traffic Accident Severity Analytics project.
--
-- Design choice: this is intentionally a light "fact + a couple of
-- supporting tables" design rather than a full star schema with many
-- dimension tables. That keeps it simple to explain while still
-- demonstrating: a raw/cleaned fact table, a related weather table,
-- a rejected-records table, and a pre-aggregated analytical mart.
--
-- Run this once to set up the database structure before load_database.py
-- inserts any data.
-- ============================================================

-- Table 1: accidents (the cleaned "fact" table - one row per accident)
CREATE TABLE IF NOT EXISTS accidents (
    accident_index          VARCHAR(50) PRIMARY KEY,
    accident_date           DATE NOT NULL,
    accident_time           VARCHAR(5),
    day_of_week             VARCHAR(15),
    accident_severity       VARCHAR(10) NOT NULL CHECK (accident_severity IN ('Fatal', 'Serious', 'Slight')),
    number_of_vehicles      INTEGER,
    number_of_casualties    INTEGER,
    road_type                VARCHAR(50),
    speed_limit              INTEGER,
    light_conditions         VARCHAR(50),
    road_surface_conditions  VARCHAR(50),
    latitude                 DOUBLE PRECISION,
    longitude                DOUBLE PRECISION,
    local_authority          VARCHAR(100),
    temperature_c            DOUBLE PRECISION,
    precipitation_mm         DOUBLE PRECISION,
    windspeed_max_kmh        DOUBLE PRECISION,
    loaded_at                TIMESTAMP DEFAULT NOW()
);

-- Table 2: weather (kept as its own table too, so weather can be queried /
-- joined independently of accidents - one row per unique location/date group)
CREATE TABLE IF NOT EXISTS weather (
    weather_id          SERIAL PRIMARY KEY,
    rounded_lat         DOUBLE PRECISION,
    rounded_lon         DOUBLE PRECISION,
    weather_date        DATE,
    temperature_c       DOUBLE PRECISION,
    precipitation_mm    DOUBLE PRECISION,
    windspeed_max_kmh   DOUBLE PRECISION,
    UNIQUE (rounded_lat, rounded_lon, weather_date)
);

-- Table 3: rejected_accidents (invalid records kept separately, with reasons)
CREATE TABLE IF NOT EXISTS rejected_accidents (
    id                   SERIAL PRIMARY KEY,
    accident_index       VARCHAR(50),
    accident_date        VARCHAR(20),
    accident_time        VARCHAR(20),
    accident_severity    VARCHAR(20),
    road_type            VARCHAR(50),
    latitude             VARCHAR(30),
    longitude            VARCHAR(30),
    rejection_reasons    TEXT,
    loaded_at            TIMESTAMP DEFAULT NOW()
);

-- Table 4: accident_analytics (the analytical data mart)
-- Pre-aggregated summary: one row per (date, road_type, severity) combination,
-- with counts ready for fast dashboard queries instead of recomputing every time.
CREATE TABLE IF NOT EXISTS accident_analytics (
    id                      SERIAL PRIMARY KEY,
    accident_date           DATE,
    road_type               VARCHAR(50),
    accident_severity       VARCHAR(10),
    total_accidents         INTEGER,
    total_vehicles_involved INTEGER,
    total_casualties        INTEGER,
    avg_temperature_c       DOUBLE PRECISION,
    avg_precipitation_mm    DOUBLE PRECISION,
    generated_at            TIMESTAMP DEFAULT NOW()
);

-- Helpful indexes for the dashboard's most common filters
CREATE INDEX IF NOT EXISTS idx_accidents_date ON accidents (accident_date);
CREATE INDEX IF NOT EXISTS idx_accidents_severity ON accidents (accident_severity);
CREATE INDEX IF NOT EXISTS idx_analytics_date ON accident_analytics (accident_date);
