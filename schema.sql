CREATE TABLE IF NOT EXISTS dim_date (
    date_id SERIAL PRIMARY KEY,
    accident_date DATE UNIQUE NOT NULL,
    day_of_week VARCHAR(20),
    day INT,
    month INT,
    year INT,
    is_weekend BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_location (
    location_id SERIAL PRIMARY KEY,
    local_authority VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS dim_road (
    road_id SERIAL PRIMARY KEY,
    road_type VARCHAR(100),
    speed_limit INT,
    light_conditions VARCHAR(100),
    road_surface_conditions VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_weather (
    weather_id SERIAL PRIMARY KEY,
    weather_code INT,
    weather_condition VARCHAR(100),
    temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    rain DOUBLE PRECISION,
    snowfall DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS dim_severity (
    severity_id SERIAL PRIMARY KEY,
    severity_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_accident (
    accident_index VARCHAR(50) PRIMARY KEY,
    date_id INT REFERENCES dim_date(date_id),
    location_id INT REFERENCES dim_location(location_id),
    road_id INT REFERENCES dim_road(road_id),
    weather_id INT REFERENCES dim_weather(weather_id),
    severity_id INT REFERENCES dim_severity(severity_id),
    accident_time VARCHAR(10),
    accident_hour INT,
    number_of_vehicles INT,
    number_of_casualties INT
);

CREATE TABLE IF NOT EXISTS accident_analytics (
    accident_index VARCHAR(50) PRIMARY KEY,
    accident_date DATE,
    accident_time VARCHAR(10),
    accident_hour INT,
    day_of_week VARCHAR(20),
    accident_severity VARCHAR(50),
    number_of_vehicles INT,
    number_of_casualties INT,
    road_type VARCHAR(100),
    speed_limit INT,
    light_conditions VARCHAR(100),
    road_surface_conditions VARCHAR(100),
    local_authority VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    weather_code INT,
    weather_condition VARCHAR(100),
    temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    rain DOUBLE PRECISION,
    snowfall DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_location_natural_key
    ON dim_location (local_authority, latitude, longitude);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_road_natural_key
    ON dim_road (road_type, speed_limit, light_conditions, road_surface_conditions);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_weather_natural_key
    ON dim_weather (weather_code, weather_condition, temperature, precipitation, rain, snowfall, wind_speed);
