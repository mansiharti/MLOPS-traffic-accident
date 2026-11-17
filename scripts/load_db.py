import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()


def get_pg_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "traffic_accidents")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "traffic_password")

    # URL.create keeps special characters in a password safely encoded.
    database_url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=dbname,
    )
    return create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def init_sqlite_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
    DROP VIEW IF EXISTS accident_analytics;
    DROP TABLE IF EXISTS accident_analytics;
    DROP TABLE IF EXISTS fact_accident;
    DROP TABLE IF EXISTS dim_severity;
    DROP TABLE IF EXISTS dim_weather;
    DROP TABLE IF EXISTS dim_road;
    DROP TABLE IF EXISTS dim_location;
    DROP TABLE IF EXISTS dim_date;

    CREATE TABLE dim_date (
        date_id INTEGER PRIMARY KEY AUTOINCREMENT,
        accident_date TEXT UNIQUE NOT NULL,
        day_of_week TEXT,
        day INTEGER,
        month INTEGER,
        year INTEGER,
        is_weekend INTEGER
    );

    CREATE TABLE dim_location (
        location_id INTEGER PRIMARY KEY AUTOINCREMENT,
        local_authority TEXT,
        latitude REAL,
        longitude REAL
    );

    CREATE TABLE dim_road (
        road_id INTEGER PRIMARY KEY AUTOINCREMENT,
        road_type TEXT,
        speed_limit INTEGER,
        light_conditions TEXT,
        road_surface_conditions TEXT
    );

    CREATE TABLE dim_weather (
        weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
        weather_code INTEGER,
        weather_condition TEXT,
        temperature REAL,
        precipitation REAL,
        rain REAL,
        snowfall REAL,
        wind_speed REAL
    );

    CREATE TABLE dim_severity (
        severity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE fact_accident (
        accident_index TEXT PRIMARY KEY,
        date_id INTEGER,
        location_id INTEGER,
        road_id INTEGER,
        weather_id INTEGER,
        severity_id INTEGER,
        accident_time TEXT,
        accident_hour INTEGER,
        number_of_vehicles INTEGER,
        number_of_casualties INTEGER,
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
        FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
        FOREIGN KEY (road_id) REFERENCES dim_road(road_id),
        FOREIGN KEY (weather_id) REFERENCES dim_weather(weather_id),
        FOREIGN KEY (severity_id) REFERENCES dim_severity(severity_id)
    );

    CREATE TABLE accident_analytics (
        accident_index TEXT PRIMARY KEY,
        accident_date TEXT,
        accident_time TEXT,
        accident_hour INTEGER,
        day_of_week TEXT,
        accident_severity TEXT,
        number_of_vehicles INTEGER,
        number_of_casualties INTEGER,
        road_type TEXT,
        speed_limit INTEGER,
        light_conditions TEXT,
        road_surface_conditions TEXT,
        local_authority TEXT,
        latitude REAL,
        longitude REAL,
        weather_code INTEGER,
        weather_condition TEXT,
        temperature REAL,
        precipitation REAL,
        rain REAL,
        snowfall REAL,
        wind_speed REAL
    );
    """)
    conn.commit()


def init_pg_schema(engine, schema_path="schema.sql"):
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            sql = f.read()
        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
        finally:
            conn.close()


def native_value(value):
    """Convert pandas/numpy scalar values to values accepted by psycopg2."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def load_dataset(input_csv="data/output/accidents_enriched.csv"):
    if not os.path.exists(input_csv):
        fallback_csv = "data/output/accidents_cleaned.csv"
        if os.path.exists(fallback_csv):
            input_csv = fallback_csv
        else:
            raise FileNotFoundError("No accident CSV found in data/output/. Run pipeline.py first.")

    df = pd.read_csv(input_csv)
    # District identifiers can look numeric; keep them as stable dimension keys.
    df["local_authority"] = df["local_authority"].astype(str)
    print(f"\n[Database Loader] Processing {len(df)} records from {input_csv}...")

    for col, default_val in [
        ("weather_code", 3),
        ("weather_condition", "Overcast"),
        ("temperature", 10.0),
        ("precipitation", 0.0),
        ("rain", 0.0),
        ("snowfall", 0.0),
        ("wind_speed", 12.0),
    ]:
        if col not in df.columns:
            df[col] = default_val

    # Initialize embedded SQLite warehouse first (ensures local availability)
    db_path = os.path.join("data", "traffic_accidents.db")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    sqlite_conn = sqlite3.connect(db_path)
    init_sqlite_schema(sqlite_conn)
    targets = [("SQLite", sqlite_conn)]

    engine = None
    pg_conn = None
    try:
        engine = get_pg_engine()
        init_pg_schema(engine)
        pg_conn = engine.raw_connection()
        reset_cursor = pg_conn.cursor()
        reset_cursor.execute(
            "TRUNCATE TABLE accident_analytics, fact_accident, dim_weather, dim_road, "
            "dim_location, dim_date, dim_severity RESTART IDENTITY CASCADE;"
        )
        pg_conn.commit()
        reset_cursor.close()
        targets.append(("PostgreSQL", pg_conn))
        print("[Database Loader] Connected to PostgreSQL warehouse!")
    except Exception as pg_err:
        print(f"[Database Loader] PostgreSQL not accessible ({pg_err}). Using embedded SQLite warehouse.")
        if engine:
            try:
                engine.dispose()
            except Exception:
                pass
        engine = None
        pg_conn = None

    for db_type, conn in targets:
        print(f"\n[Database Loader] Populating {db_type} tables...")
        cur = conn.cursor()

        severities = ["Fatal", "Serious", "Slight"]
        for sev in severities:
            if db_type == "PostgreSQL":
                cur.execute(
                    "INSERT INTO dim_severity (severity_name) VALUES (%s) ON CONFLICT (severity_name) DO NOTHING;",
                    (sev,),
                )
            else:
                cur.execute("INSERT OR IGNORE INTO dim_severity (severity_name) VALUES (?);", (sev,))

        dates_df = df[["accident_date", "day_of_week"]].drop_duplicates().copy()
        dates_df["dt"] = pd.to_datetime(dates_df["accident_date"])
        dates_df["day"] = dates_df["dt"].dt.day
        dates_df["month"] = dates_df["dt"].dt.month
        dates_df["year"] = dates_df["dt"].dt.year
        dates_df["is_weekend"] = dates_df["dt"].dt.dayofweek.isin([5, 6]).astype(int)

        for _, r in dates_df.iterrows():
            if db_type == "PostgreSQL":
                cur.execute(
                    """INSERT INTO dim_date (accident_date, day_of_week, day, month, year, is_weekend)
                       VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (accident_date) DO NOTHING;""",
                    (r["accident_date"], r["day_of_week"], int(r["day"]), int(r["month"]), int(r["year"]), bool(r["is_weekend"])),
                )
            else:
                cur.execute(
                    """INSERT OR IGNORE INTO dim_date (accident_date, day_of_week, day, month, year, is_weekend)
                       VALUES (?, ?, ?, ?, ?, ?);""",
                    (r["accident_date"], r["day_of_week"], int(r["day"]), int(r["month"]), int(r["year"]), int(r["is_weekend"])),
                )

        loc_df = df[["local_authority", "latitude", "longitude"]].drop_duplicates()
        loc_tuples = [(str(r["local_authority"]), float(r["latitude"]), float(r["longitude"])) for _, r in loc_df.iterrows()]
        if db_type == "PostgreSQL":
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """INSERT INTO dim_location (local_authority, latitude, longitude)
                   VALUES %s
                   ON CONFLICT (local_authority, latitude, longitude) DO NOTHING;""",
                loc_tuples,
            )
        else:
            cur.executemany(
                """INSERT INTO dim_location (local_authority, latitude, longitude)
                   VALUES (?, ?, ?);""",
                loc_tuples,
            )

        road_df = df[["road_type", "speed_limit", "light_conditions", "road_surface_conditions"]].drop_duplicates()
        for _, r in road_df.iterrows():
            if db_type == "PostgreSQL":
                cur.execute(
                    """INSERT INTO dim_road (road_type, speed_limit, light_conditions, road_surface_conditions)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (road_type, speed_limit, light_conditions, road_surface_conditions) DO NOTHING;""",
                    (str(r["road_type"]), int(r["speed_limit"]), str(r["light_conditions"]), str(r["road_surface_conditions"])),
                )
            else:
                cur.execute(
                    """INSERT INTO dim_road (road_type, speed_limit, light_conditions, road_surface_conditions)
                       VALUES (?, ?, ?, ?);""",
                    (r["road_type"], int(r["speed_limit"]), r["light_conditions"], r["road_surface_conditions"]),
                )

        weather_cols = ["weather_code", "weather_condition", "temperature", "precipitation", "rain", "snowfall", "wind_speed"]
        weath_df = df[weather_cols].drop_duplicates()
        for _, r in weath_df.iterrows():
            if db_type == "PostgreSQL":
                cur.execute(
                    """INSERT INTO dim_weather (weather_code, weather_condition, temperature, precipitation, rain, snowfall, wind_speed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (weather_code, weather_condition, temperature, precipitation, rain, snowfall, wind_speed) DO NOTHING;""",
                    (int(r["weather_code"]), r["weather_condition"], float(r["temperature"]), float(r["precipitation"]), float(r["rain"]), float(r["snowfall"]), float(r["wind_speed"])),
                )
            else:
                cur.execute(
                    """INSERT INTO dim_weather (weather_code, weather_condition, temperature, precipitation, rain, snowfall, wind_speed)
                       VALUES (?, ?, ?, ?, ?, ?, ?);""",
                    (int(r["weather_code"]), r["weather_condition"], float(r["temperature"]), float(r["precipitation"]), float(r["rain"]), float(r["snowfall"]), float(r["wind_speed"])),
                )

        conn.commit()

        cur.execute("SELECT accident_date, date_id FROM dim_date;")
        date_map = {str(accident_date): date_id for accident_date, date_id in cur.fetchall()}
        cur.execute("SELECT severity_name, severity_id FROM dim_severity;")
        sev_map = dict(cur.fetchall())

        loc_lookup = {}
        cur.execute("SELECT location_id, local_authority, latitude, longitude FROM dim_location;")
        for row in cur.fetchall():
            loc_lookup[(row[1], round(row[2], 4), round(row[3], 4))] = row[0]

        road_lookup = {}
        cur.execute("SELECT road_id, road_type, speed_limit, light_conditions, road_surface_conditions FROM dim_road;")
        for row in cur.fetchall():
            road_lookup[(row[1], row[2], row[3], row[4])] = row[0]

        weath_lookup = {}
        cur.execute("SELECT weather_id, weather_code, weather_condition, temperature, precipitation, rain, snowfall, wind_speed FROM dim_weather;")
        for row in cur.fetchall():
            weath_lookup[(row[1], row[2], round(row[3], 1), round(row[4], 2), round(row[5], 2), round(row[6], 2), round(row[7], 1))] = row[0]

        fact_rows = []
        for _, r in df.iterrows():
            d_id = date_map.get(r["accident_date"])
            s_id = sev_map.get(r["accident_severity"])
            l_id = loc_lookup.get((r["local_authority"], round(float(r["latitude"]), 4), round(float(r["longitude"]), 4)))
            rd_id = road_lookup.get((r["road_type"], int(r["speed_limit"]), r["light_conditions"], r["road_surface_conditions"]))
            w_id = weath_lookup.get((
                int(r["weather_code"]),
                r["weather_condition"],
                round(float(r["temperature"]), 1),
                round(float(r["precipitation"]), 2),
                round(float(r["rain"]), 2),
                round(float(r["snowfall"]), 2),
                round(float(r["wind_speed"]), 1),
            ))

            fact_rows.append((
                str(r["accident_index"]),
                d_id,
                l_id,
                rd_id,
                w_id,
                s_id,
                str(r["accident_time"]),
                int(r["accident_hour"]),
                int(r["number_of_vehicles"]),
                int(r["number_of_casualties"]),
            ))

        if db_type == "PostgreSQL":
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """INSERT INTO fact_accident (accident_index, date_id, location_id, road_id, weather_id, severity_id,
                                              accident_time, accident_hour, number_of_vehicles, number_of_casualties)
                   VALUES %s ON CONFLICT (accident_index) DO NOTHING;""",
                fact_rows,
            )
        else:
            cur.executemany(
                """INSERT OR IGNORE INTO fact_accident (accident_index, date_id, location_id, road_id, weather_id, severity_id,
                                                        accident_time, accident_hour, number_of_vehicles, number_of_casualties)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                fact_rows,
            )

        mart_cols = [
            "accident_index", "accident_date", "accident_time", "accident_hour", "day_of_week",
            "accident_severity", "number_of_vehicles", "number_of_casualties", "road_type",
            "speed_limit", "light_conditions", "road_surface_conditions", "local_authority",
            "latitude", "longitude", "weather_code", "weather_condition", "temperature",
            "precipitation", "rain", "snowfall", "wind_speed"
        ]
        mart_rows = [tuple(native_value(row[c]) for c in mart_cols) for _, row in df.iterrows()]

        if db_type == "PostgreSQL":
            from psycopg2.extras import execute_values
            placeholders = ", ".join(mart_cols)
            execute_values(
                cur,
                f"""INSERT INTO accident_analytics ({placeholders})
                    VALUES %s ON CONFLICT (accident_index) DO NOTHING;""",
                mart_rows,
            )
        else:
            q_marks = ", ".join(["?"] * len(mart_cols))
            placeholders = ", ".join(mart_cols)
            cur.executemany(
                f"""INSERT OR IGNORE INTO accident_analytics ({placeholders})
                    VALUES ({q_marks});""",
                mart_rows,
            )

        conn.commit()

        print("  -> Verification Table Counts:")
        for tbl in ["dim_date", "dim_location", "dim_road", "dim_weather", "dim_severity", "fact_accident", "accident_analytics"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            cnt = cur.fetchone()[0]
            print(f"     - {tbl.ljust(20)}: {cnt} records")

        conn.close()

    if engine:
        try:
            engine.dispose()
        except Exception:
            pass

    print("\n[Database Loader] Finished successfully! Analytical warehouse ready.")


if __name__ == "__main__":
    load_dataset()
