"""
app.py
------
Streamlit dashboard for the Traffic Accident Severity Analytics project.

WHAT THIS DOES:
Connects to the PostgreSQL database and shows the required views:
  1. Total number of accidents
  2. Accident severity distribution
  3. Accidents by hour of day
  4. Accidents by weekday
  5. Accidents by road type
  6. Weather-condition analysis
  7. Accident location/hotspot map
  8. (Optional) Vehicle/casualty analysis

WHY WE QUERY THE `accidents` TABLE DIRECTLY FOR MOST CHARTS, BUT USE
`accident_analytics` FOR THE ROAD-TYPE VIEW:
Most of these charts need to compute things (hour of day, weekday) that
weren't pre-aggregated in the mart, so it's simplest and clear enough for
this project size to query `accidents` directly with pandas for those.
The road-type-by-severity breakdown, though, maps directly onto the
pre-built `accident_analytics` mart, so that view demonstrates the mart
being used, which is one of the assignment requirements.

HOW TO RUN:
    streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# Let this file import config.py from the sibling scripts/ folder
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from config import DB_CONNECTION_STRING

st.set_page_config(page_title="Traffic Accident Severity Analytics", layout="wide")


@st.cache_data(ttl=300)
def load_accidents():
    engine = create_engine(DB_CONNECTION_STRING)
    df = pd.read_sql("SELECT * FROM accidents", engine)
    df["accident_date"] = pd.to_datetime(df["accident_date"])
    df["hour"] = pd.to_datetime(df["accident_time"], format="%H:%M", errors="coerce").dt.hour
    return df


@st.cache_data(ttl=300)
def load_analytics_mart():
    engine = create_engine(DB_CONNECTION_STRING)
    return pd.read_sql("SELECT * FROM accident_analytics", engine)


st.title("🚦 Traffic Accident Severity Analytics")
st.caption("Part 1 - Data Engineering pipeline output, visualised from PostgreSQL")

try:
    accidents_df = load_accidents()
    mart_df = load_analytics_mart()
except Exception as error:
    st.error(
        "Could not connect to the database. Make sure PostgreSQL is running "
        "and load_database.py has been run at least once.\n\n"
        f"Technical detail: {error}"
    )
    st.stop()

# ---------------------------------------------------------------
# Sidebar filters (simple - keeps the rest of the dashboard focused)
# ---------------------------------------------------------------
st.sidebar.header("Filters")
severity_filter = st.sidebar.multiselect(
    "Severity", options=sorted(accidents_df["accident_severity"].unique()),
    default=sorted(accidents_df["accident_severity"].unique()),
)
filtered_df = accidents_df[accidents_df["accident_severity"].isin(severity_filter)]

# ---------------------------------------------------------------
# View 1: Total number of accidents
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Accidents", len(filtered_df))
col2.metric("Total Casualties", int(filtered_df["number_of_casualties"].sum()))
col3.metric("Total Vehicles Involved", int(filtered_df["number_of_vehicles"].sum()))

st.divider()

# ---------------------------------------------------------------
# View 2: Accident severity distribution
# ---------------------------------------------------------------
st.subheader("Accident Severity Distribution")
severity_counts = filtered_df["accident_severity"].value_counts()
st.bar_chart(severity_counts)

# ---------------------------------------------------------------
# View 3 & 4: Accidents by hour and by weekday, side by side
# ---------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Accidents by Hour of Day")
    by_hour = filtered_df.groupby("hour").size().sort_index()
    st.bar_chart(by_hour)

with col_b:
    st.subheader("Accidents by Weekday")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_weekday = filtered_df.groupby("day_of_week").size().reindex(weekday_order)
    st.bar_chart(by_weekday)

# ---------------------------------------------------------------
# View 5: Accidents by road type (using the analytics mart table)
# ---------------------------------------------------------------
st.subheader("Accidents by Road Type (from the analytics data mart)")
road_type_summary = (
    mart_df.groupby("road_type")["total_accidents"].sum().sort_values(ascending=False)
)
st.bar_chart(road_type_summary)

# ---------------------------------------------------------------
# View 6: Weather-condition analysis
# ---------------------------------------------------------------
st.subheader("Weather Condition Analysis")
weather_available = filtered_df["temperature_c"].notna().sum() > 0

if weather_available:
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("**Average Temperature by Severity (°C)**")
        st.bar_chart(filtered_df.groupby("accident_severity")["temperature_c"].mean())
    with col_w2:
        st.markdown("**Average Precipitation by Severity (mm)**")
        st.bar_chart(filtered_df.groupby("accident_severity")["precipitation_mm"].mean())
else:
    st.info(
        "No weather data available yet. This happens if weather.py could not "
        "reach the Open-Meteo API (e.g. no internet connection when it ran). "
        "Re-run weather.py with an internet connection, then load_database.py."
    )

# ---------------------------------------------------------------
# View 7: Accident location / hotspot map
# ---------------------------------------------------------------
st.subheader("Accident Hotspot Map")
map_df = filtered_df[["latitude", "longitude"]].dropna()
if not map_df.empty:
    st.map(map_df, latitude="latitude", longitude="longitude", size=20)
else:
    st.info("No valid coordinates available to plot.")

# ---------------------------------------------------------------
# View 8 (Optional): Vehicle / casualty analysis
# ---------------------------------------------------------------
st.subheader("Vehicle & Casualty Analysis (Optional View)")
col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("**Average Vehicles Involved by Severity**")
    st.bar_chart(filtered_df.groupby("accident_severity")["number_of_vehicles"].mean())
with col_v2:
    st.markdown("**Average Casualties by Severity**")
    st.bar_chart(filtered_df.groupby("accident_severity")["number_of_casualties"].mean())

st.divider()
st.caption("Data source: UK STATS19-style accident data (sample/demo data - "
           "swap in the real download for production use) + Open-Meteo historical weather API.")
