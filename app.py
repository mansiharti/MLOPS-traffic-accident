import os
import json
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Traffic Accident Severity Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


def filter_by_location(data, search):
    query = search.strip()
    if not query:
        return data
    searchable_columns = ["local_authority", "local_authority_name", "latitude", "longitude"]
    searchable = pd.Series(False, index=data.index)
    for column in searchable_columns:
        if column in data.columns:
            searchable |= data[column].astype("string").str.contains(query, case=False, na=False, regex=False)
    return data[searchable]


@st.cache_data(ttl=3600)
def load_authority_names():
    path = os.path.join("data", "authority_names.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as names_file:
        return json.load(names_file)


def make_condition_report(data):
    rows = []
    conditions = {"Weather": "weather_condition", "Road surface": "road_surface_conditions", "Lighting": "light_conditions", "Road type": "road_type", "Speed limit": "speed_limit"}
    for name, column in conditions.items():
        if column in data:
            counts = data[column].fillna("Unknown").replace("", "Unknown").value_counts()
            rows += [{"Condition": name, "Recorded value": value, "Accidents": int(count), "Percent": round(count / len(data) * 100, 1)} for value, count in counts.items()]
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def load_dataset():
    db_path = os.path.join("data", "traffic_accidents.db")
    enriched_csv = os.path.join("data", "output", "accidents_enriched.csv")
    cleaned_csv = os.path.join("data", "output", "accidents_cleaned.csv")
    rejected_csv = os.path.join("data", "output", "rejected_records.csv")

    df = None
    source_name = "None"

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "traffic_accidents"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "traffic_password"),
            connect_timeout=3,
        )
        df = pd.read_sql_query("SELECT * FROM accident_analytics;", conn)
        conn.close()
        source_name = "PostgreSQL Database (accident_analytics)"
    except Exception:
        df = None

    # SQLite remains a fallback for offline use. Prefer PostgreSQL whenever it
    # is available so the dashboard shows the Docker warehouse.
    if df is None and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM accident_analytics;", conn)
            conn.close()
            source_name = "SQLite Database (accident_analytics)"
        except Exception:
            df = None

    if df is None or df.empty:
        if os.path.exists(enriched_csv):
            df = pd.read_csv(enriched_csv)
            source_name = "CSV File (accidents_enriched.csv)"
        elif os.path.exists(cleaned_csv):
            df = pd.read_csv(cleaned_csv)
            source_name = "CSV File (accidents_cleaned.csv)"

    rejected_df = pd.DataFrame()
    if os.path.exists(rejected_csv):
        try:
            rejected_df = pd.read_csv(rejected_csv)
        except Exception:
            pass

    if df is not None:
        authority_names = load_authority_names()
        df["local_authority_name"] = df["local_authority"].astype(str).map(authority_names).fillna(df["local_authority"])
        df["accident_date"] = pd.to_datetime(df["accident_date"])
        df["month_year"] = df["accident_date"].dt.to_period("M").astype(str)

    return df, rejected_df, source_name


df, df_rejected, data_source = load_dataset()

if df is None or df.empty:
    st.error("No dataset found. Please run 'python main.py' to generate the data.")
    st.stop()

st.title("Traffic Accident Severity Analytics")
st.caption(f"Data source: {data_source} | UK Road Safety records enriched with Open-Meteo historical weather")
st.markdown("---")

with st.sidebar:
    st.header("Filters")

    min_date = df["accident_date"].min().date()
    max_date = df["accident_date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_severities = sorted(df["accident_severity"].unique().tolist())
    selected_severities = st.multiselect(
        "Accident Severity",
        options=all_severities,
        default=all_severities,
    )

    authorities = ["All"] + sorted(df["local_authority_name"].dropna().unique().tolist())
    selected_authority = st.selectbox("Local Authority", options=authorities)

    road_types = ["All"] + sorted(df["road_type"].dropna().unique().tolist())
    selected_road_type = st.selectbox("Road Type", options=road_types)

    if "weather_condition" in df.columns:
        weather_options = ["All"] + sorted(df["weather_condition"].dropna().unique().tolist())
        selected_weather = st.selectbox("Weather Condition", options=weather_options)
    else:
        selected_weather = "All"

    st.markdown("---")
    st.markdown(f"Total Clean Records: **{len(df):,}**")
    st.markdown(f"Total Rejected Records: **{len(df_rejected):,}**")

filtered = df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered = filtered[
        (filtered["accident_date"].dt.date >= start_d) &
        (filtered["accident_date"].dt.date <= end_d)
    ]

if selected_severities:
    filtered = filtered[filtered["accident_severity"].isin(selected_severities)]

if selected_authority != "All":
    filtered = filtered[filtered["local_authority_name"] == selected_authority]

if selected_road_type != "All":
    filtered = filtered[filtered["road_type"] == selected_road_type]

if selected_weather != "All" and "weather_condition" in filtered.columns:
    filtered = filtered[filtered["weather_condition"] == selected_weather]

if filtered.empty:
    st.warning("No records match the selected filters. Please adjust your criteria.")
    st.stop()

total_accidents = len(filtered)
total_casualties = int(filtered["number_of_casualties"].sum())
total_vehicles = int(filtered["number_of_vehicles"].sum())
fatal_count = int((filtered["accident_severity"] == "Fatal").sum())
serious_count = int((filtered["accident_severity"] == "Serious").sum())
slight_count = int((filtered["accident_severity"] == "Slight").sum())
fatality_rate = (fatal_count / total_accidents * 100) if total_accidents > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Accidents", f"{total_accidents:,}")
col2.metric("Total Casualties", f"{total_casualties:,}", f"{(total_casualties/total_accidents):.2f} per accident")
col3.metric("Fatal Accidents", f"{fatal_count}", f"{fatality_rate:.1f}% of total")
col4.metric("Serious Accidents", f"{serious_count}", f"{(serious_count/total_accidents*100):.1f}% of total")
col5.metric("Slight Accidents", f"{slight_count}", f"{(slight_count/total_accidents*100):.1f}% of total")

st.markdown("---")

SEVERITY_COLORS = {
    "Fatal": "#dc2626",
    "Serious": "#d97706",
    "Slight": "#16a34a",
}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Severity & Temporal Patterns",
    "2. Location Hotspot Map",
    "3. Weather & Road Conditions",
    "4. Vehicle & Casualty Analysis",
    "5. Data Mart Explorer",
    "6. ML Severity Prediction & MLOps Monitoring",
])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Severity Distribution")
        sev_df = filtered["accident_severity"].value_counts().reset_index()
        sev_df.columns = ["Severity", "Count"]

        fig_pie = px.pie(
            sev_df,
            values="Count",
            names="Severity",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Accidents by Hour of Day")
        hourly = filtered.groupby(["accident_hour", "accident_severity"]).size().reset_index(name="Accidents")
        fig_hour = px.bar(
            hourly,
            x="accident_hour",
            y="Accidents",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"accident_hour": "Hour (0-23)", "Accidents": "Accident Count"},
        )
        fig_hour.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Monthly Accident Volume Trend")
        monthly = filtered.groupby("month_year").agg(
            Accidents=("accident_index", "count"),
            Casualties=("number_of_casualties", "sum"),
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=monthly["month_year"], y=monthly["Accidents"], name="Accidents", marker_color="#2563eb"))
        fig_trend.add_trace(go.Scatter(x=monthly["month_year"], y=monthly["Casualties"], name="Casualties", line=dict(color="#dc2626", width=2), yaxis="y2"))
        fig_trend.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(title="Accidents"),
            yaxis2=dict(title="Casualties", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_d:
        st.subheader("Accidents by Day of Week")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = filtered.groupby("day_of_week").size().reindex(days).fillna(0).reset_index()
        dow_df.columns = ["Day", "Accidents"]

        fig_dow = px.bar(
            dow_df,
            x="Day",
            y="Accidents",
            color="Accidents",
            color_continuous_scale="Blues",
            text="Accidents",
        )
        fig_dow.update_traces(textposition="outside")
        fig_dow.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

with tab2:
    st.subheader("Geographic Collision Hotspots")

    location_search = st.text_input(
        "Search location or local authority:",
        placeholder="e.g. Birmingham, Leeds, Sheffield",
        key="location_hotspot_search",
    )
    available_locations = sorted(filtered["local_authority_name"].dropna().astype(str).unique())
    st.caption(f"Available locations: {', '.join(available_locations)}")
    map_filtered = filter_by_location(filtered, location_search)

    if map_filtered.empty:
        st.info(
            f"No accident records found for '{location_search.strip()}'. "
            "Try another location or local authority."
        )
    else:
        st.caption(f"Showing {len(map_filtered):,} records for the selected location search.")

    if not map_filtered.empty:
        map_kwargs = dict(
            lat="latitude",
            lon="longitude",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            size="number_of_casualties",
            size_max=12,
            hover_name="local_authority_name",
            hover_data={"accident_date": True, "speed_limit": True, "road_type": True, "latitude": False, "longitude": False},
            zoom=5.5,
            center=dict(lat=53.4, lon=-1.8),
        )
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(map_filtered, map_style="carto-positron", **map_kwargs)
        else:
            fig_map = px.scatter_mapbox(map_filtered, mapbox_style="carto-positron", **map_kwargs)

        fig_map.update_layout(height=480, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Accidents by Local Authority")
        auth_counts = map_filtered.groupby("local_authority_name").size().reset_index(name="Accidents").sort_values(by="Accidents", ascending=True)
        fig_auth = px.bar(auth_counts, x="Accidents", y="local_authority_name", orientation="h", text="Accidents")
        fig_auth.update_traces(textposition="outside")
        fig_auth.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(title=""))
        st.plotly_chart(fig_auth, use_container_width=True)

        st.subheader("Accident Reasons and Area Report")
        st.caption(
            "The source dataset does not contain a confirmed cause field. "
            "These are recorded conditions associated with accidents in the selected area."
        )

        condition_report = make_condition_report(map_filtered)
        st.dataframe(condition_report, use_container_width=True, hide_index=True)

        report_col1, report_col2 = st.columns(2)
        with report_col1:
            summary_csv = condition_report.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Conditions Report",
                data=summary_csv,
                file_name="location_accident_conditions_report.csv",
                mime="text/csv",
                key="download_conditions_report",
            )
        with report_col2:
            detail_csv = map_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Area Accident Records",
                data=detail_csv,
                file_name="location_accident_records.csv",
                mime="text/csv",
                key="download_area_accident_records",
            )

with tab3:
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.subheader("Accidents by Weather Condition")
        if "weather_condition" in filtered.columns:
            weath = filtered.groupby(["weather_condition", "accident_severity"]).size().reset_index(name="Accidents")
            fig_w = px.bar(
                weath,
                x="weather_condition",
                y="Accidents",
                color="accident_severity",
                color_discrete_map=SEVERITY_COLORS,
                barmode="stack",
            )
            fig_w.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(tickangle=-30))
            st.plotly_chart(fig_w, use_container_width=True)

    with col_w2:
        st.subheader("Accidents by Road Surface Condition")
        surf = filtered.groupby(["road_surface_conditions", "accident_severity"]).size().reset_index(name="Accidents")
        fig_surf = px.bar(
            surf,
            x="road_surface_conditions",
            y="Accidents",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
        )
        fig_surf.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_surf, use_container_width=True)

    col_w3, col_w4 = st.columns(2)

    with col_w3:
        st.subheader("Ambient Temperature (C) Distribution")
        if "temperature" in filtered.columns:
            fig_t = px.histogram(filtered, x="temperature", color="accident_severity", color_discrete_map=SEVERITY_COLORS, nbins=20)
            fig_t.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_t, use_container_width=True)

    with col_w4:
        st.subheader("Wind Speed (km/h) by Severity")
        if "wind_speed" in filtered.columns:
            fig_wind = px.box(filtered, x="accident_severity", y="wind_speed", color="accident_severity", color_discrete_map=SEVERITY_COLORS)
            fig_wind.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_wind, use_container_width=True)

with tab4:
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Casualties by Speed Limit")
        speed_cas = filtered.groupby(["speed_limit", "accident_severity"])["number_of_casualties"].sum().reset_index()
        fig_spd = px.bar(
            speed_cas,
            x="speed_limit",
            y="number_of_casualties",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"speed_limit": "Speed Limit (mph)", "number_of_casualties": "Total Casualties"},
        )
        fig_spd.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_spd, use_container_width=True)

    with col_v2:
        st.subheader("Vehicles Involved Distribution")
        veh_counts = filtered.groupby(["number_of_vehicles", "accident_severity"]).size().reset_index(name="Accidents")
        fig_veh = px.bar(
            veh_counts,
            x="number_of_vehicles",
            y="Accidents",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"number_of_vehicles": "Number of Vehicles", "Accidents": "Accident Count"},
        )
        fig_veh.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_veh, use_container_width=True)

    st.subheader("Data Quality & Quarantine Ingestion Summary")
    qa_c1, qa_c2, qa_c3, qa_c4 = st.columns(4)
    total_in = len(df) + len(df_rejected)
    qa_c1.metric("Raw Inputs Ingested", f"{total_in:,}")
    qa_c2.metric("Valid Clean Records", f"{len(df):,}")
    qa_c3.metric("Quarantined Records", f"{len(df_rejected):,}")
    q_rate = (len(df) / total_in * 100) if total_in > 0 else 0
    qa_c4.metric("Validation Pass Rate", f"{q_rate:.1f}%")

    if not df_rejected.empty and "rejection_reason" in df_rejected.columns:
        st.caption("Quarantined records failure reasons breakdown:")
        reasons = df_rejected["rejection_reason"].str.split("; ").explode().value_counts().reset_index()
        reasons.columns = ["Failure Reason", "Count"]
        st.dataframe(reasons, use_container_width=True)

with tab5:
    st.subheader("Analytical Mart Records")
    st.caption(f"Showing {len(filtered):,} records based on active filters.")

    search = st.text_input("Search records by Accident Index, City, or Road Type:", "")
    view_data = filtered
    if search:
        mask = (
            view_data["accident_index"].astype(str).str.contains(search, case=False, na=False) |
            view_data["local_authority"].astype(str).str.contains(search, case=False, na=False) |
            view_data["road_type"].astype(str).str.contains(search, case=False, na=False)
        )
        view_data = view_data[mask]

    st.dataframe(view_data, use_container_width=True, height=400)

    csv = view_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="traffic_accidents_export.csv",
        mime="text/csv",
    )

with tab6:
    st.subheader("Machine Learning Severity Prediction & MLOps Operations")
    st.caption("Operationalized Champion Model with Real-Time Inference, MLflow Registry & Drift Monitoring")

    pred_subtab1, pred_subtab2, pred_subtab3 = st.tabs([
        "Live Risk Inference",
        "MLflow Model Registry & Evaluation",
        "Data Drift & Health Monitoring",
    ])

    with pred_subtab1:
        st.markdown("##### Collision Scenario Simulation & Real-Time Severity Classification")
        st.info("Adjust the incident parameters below to simulate a collision scenario. The deployed Champion Model predicts severity and generates risk alerts.")

        with st.form("severity_prediction_form"):
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.markdown("**1. Road & Location Attributes**")
                form_speed = st.selectbox("Posted Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
                form_road = st.selectbox("Road Type", ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"])
                form_light = st.selectbox("Lighting Conditions", ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit"])
                form_surface = st.selectbox("Road Surface Condition", ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm deep"])

            with c_f2:
                st.markdown("**2. Incident & Temporal Factors**")
                form_hour = st.slider("Hour of Day (0-23)", 0, 23, 18)
                form_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
                form_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=20, value=2)
                form_casualties = st.number_input("Number of Casualties", min_value=1, max_value=20, value=1)

            with c_f3:
                st.markdown("**3. Weather Telemetry**")
                form_weather = st.selectbox("Weather Condition", [
                    "Fine no high winds", "Raining no high winds", "Raining with high winds",
                    "Fine with high winds", "Fog or mist", "Snowing no high winds"
                ])
                form_temp = st.slider("Temperature (°C)", -10.0, 35.0, 10.0, step=0.5)
                form_precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                form_wind = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0, step=1.0)

            submitted = st.form_submit_button("Predict Collision Severity Risk", type="primary", use_container_width=True)

        if submitted:
            try:
                from src.predict import get_predictor
                from src.monitoring import get_monitor

                predictor = get_predictor()
                monitor = get_monitor()

                input_dict = {
                    "accident_hour": form_hour,
                    "day_of_week": form_day,
                    "speed_limit": form_speed,
                    "number_of_vehicles": form_vehicles,
                    "number_of_casualties": form_casualties,
                    "road_type": form_road,
                    "light_conditions": form_light,
                    "road_surface_conditions": form_surface,
                    "weather_condition": form_weather,
                    "temperature": form_temp,
                    "precipitation": form_precip,
                    "wind_speed": form_wind,
                }

                res = predictor.predict(input_dict)
                monitor.log_inference_event(input_dict, res)

                st.markdown("---")
                st.markdown("#### Real-Time Inference Results")

                res_c1, res_c2, res_c3, res_c4 = st.columns(4)

                severity_label = res["predicted_severity"]
                risk_alert = res["risk_alert_level"]
                confidence = res["confidence"] * 100
                latency = res["latency_ms"]

                res_c1.metric("Predicted Severity", severity_label)
                res_c2.metric("Risk Alert Level", risk_alert)
                res_c3.metric("Prediction Confidence", f"{confidence:.1f}%")
                res_c4.metric("Inference Latency", f"{latency:.1f} ms")

                probs_df = pd.DataFrame([
                    {"Severity Class": k, "Probability (%)": v * 100}
                    for k, v in res["probabilities"].items()
                ])

                fig_prob = px.bar(
                    probs_df,
                    x="Probability (%)",
                    y="Severity Class",
                    orientation="h",
                    color="Severity Class",
                    color_discrete_map=SEVERITY_COLORS,
                    text="Probability (%)",
                    title="Class Probability Distribution",
                )
                fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_prob.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=20))
                st.plotly_chart(fig_prob, use_container_width=True)

                if "CRITICAL" in risk_alert:
                    st.error("🚨 CRITICAL RISK ALERT: High probability of severe injury or fatality. Emergency dispatch prioritisation recommended.")
                elif "ELEVATED" in risk_alert:
                    st.warning("⚠️ ELEVATED RISK WARNING: Substantial probability of serious injury. Caution advised.")
                else:
                    st.success("✅ STANDARD RISK: Baseline probability distribution aligns with minor / slight impact profile.")

            except Exception as pred_err:
                st.error(f"Prediction Error: {pred_err}")

    with pred_subtab2:
        st.markdown("##### MLflow Model Registry, Parameters & Comparative Evaluation")
        meta_file = os.path.join("models", "model_metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            reg_c1, reg_c2, reg_c3, reg_c4 = st.columns(4)
            reg_c1.metric("Champion Model", metadata.get("model_name", "N/A"))
            reg_c2.metric("Model Version", metadata.get("model_version", "1.0.0"))
            reg_c3.metric("Test Macro-F1", f"{metadata['test_metrics']['test_macro_f1']:.4f}")
            reg_c4.metric("Fatal Recall (Alert)", f"{metadata['test_metrics']['test_fatal_recall']:.2%}")

            st.markdown("---")
            st.markdown("##### Model Candidate Comparison (Validation Set)")
            all_models = metadata.get("all_model_comparison", {})
            if all_models:
                comp_display = []
                for m_name, m_vals in all_models.items():
                    comp_display.append({
                        "Model Candidate": m_name,
                        "Macro-F1": f"{m_vals['val_macro_f1']:.4f}",
                        "Accuracy": f"{m_vals['val_accuracy']:.2%}",
                        "Fatal Recall": f"{m_vals['val_fatal_recall']:.2%}",
                        "Serious Recall": f"{m_vals['val_serious_recall']:.2%}",
                        "Fatal False-Negative Rate": f"{m_vals['fatal_fnr']:.2%}",
                        "Training Time (s)": f"{m_vals['train_duration_sec']:.2f}",
                    })
                st.table(pd.DataFrame(comp_display))

            cm_img_path = os.path.join("models", "artifacts", "champion_test_cm.png")
            if os.path.exists(cm_img_path):
                st.markdown("##### Champion Normalized Confusion Matrix (Unseen Chronological Test Set)")
                st.image(cm_img_path, caption="Random Forest Test Confusion Matrix (Balanced Weights)", width=480)
        else:
            st.warning("Model metadata manifest not found. Run 'python src/train.py' to generate artifacts.")

    with pred_subtab3:
        st.markdown("##### Continuous MLOps Drift Detection & Operational Retraining Health")

        top_col1, top_col2 = st.columns([3, 1])
        with top_col2:
            if st.button("🔄 Refresh Live Metrics", use_container_width=True):
                st.rerun()

        from src.monitoring import get_monitor
        monitor = get_monitor()
        metrics = monitor.get_summary_metrics()

        total_inferences = metrics["total_inferences"]
        avg_lat = metrics["avg_latency_ms"]
        psi_val = metrics["class_drift"]["psi"]
        psi_display = metrics["class_drift"]["display_text"]
        op_health = metrics["retraining_decision"]["status"]

        with top_col1:
            st.caption(
                f"**Monitoring Storage:** SQLite `inference_logs` | "
                f"**Samples Used:** {total_inferences:,} | "
                f"**Latest Activity:** {metrics['latest_timestamp'] or 'No prediction activity recorded yet'}"
            )

        mon_c1, mon_c2, mon_c3, mon_c4 = st.columns(4)
        mon_c1.metric(
            "Live API Inferences",
            f"{total_inferences:,} records" if total_inferences > 0 else "0 records"
        )
        mon_c2.metric(
            "Mean Latency",
            f"{avg_lat:.1f} ms" if avg_lat is not None else "No data yet"
        )
        mon_c3.metric(
            "Class Drift PSI",
            f"{psi_val:.4f}" if psi_val is not None else psi_display
        )
        mon_c4.metric(
            "Operational Health",
            op_health
        )

        st.markdown("---")
        st.markdown("##### Automated Retraining Criteria & Trigger Status")

        # Dynamic Retraining Criteria Table
        if psi_val is None:
            psi_status = f"**{metrics['class_drift']['status']}**"
        elif psi_val < 0.10:
            psi_status = "**STABLE**"
        elif psi_val < 0.20:
            psi_status = "**MODERATE**"
        else:
            psi_status = "**DRIFT DETECTED**"

        fatal_fnr_display = metrics["fatal_fnr"]["display_text"]
        fatal_fnr_status = f"**{metrics['fatal_fnr']['status']}**"

        dq_display = metrics["data_quality"]["display_text"]
        dq_status = f"**{metrics['data_quality']['status']}**"

        acc_display = f"{total_inferences:,} records"
        acc_status = "**MILESTONE REACHED**" if total_inferences >= 10000 else "**ACCUMULATING**"

        st.markdown(r"""
| Retraining Criterion | Threshold | Current Value | Status |
| :--- | :--- | :--- | :--- |
| **Class Drift PSI** | $\ge 0.20$ | {psi} | {psi_status} |
| **Fatal False-Negative Rate** | $\ge 60.0\%$ | {fatal_fnr} | {fatal_fnr_status} |
| **Input Data Quality Score** | $< 85.0\%$ | {dq} | {dq_status} |
| **Inference Sample Accumulation** | $\ge 10,000$ | {inferences} | {acc_status} |
| **Retraining Cadence** | Quarterly | Model Version 1.0.0 (Fresh) | **ACTIVE** |
""".format(
            psi=f"{psi_val:.4f}" if psi_val is not None else psi_display,
            psi_status=psi_status,
            fatal_fnr=fatal_fnr_display,
            fatal_fnr_status=fatal_fnr_status,
            dq=dq_display,
            dq_status=dq_status,
            inferences=acc_display,
            acc_status=acc_status,
        ))

        if total_inferences == 0:
            st.info(
                "ℹ️ **No live API predictions recorded yet.**\n\n"
                "To populate live monitoring metrics:\n"
                "- Send prediction requests to FastAPI `POST http://127.0.0.1:8000/predict` or `/predict/batch`.\n"
                "- Or use the **Live Risk Inference** simulator in Subtab 1 to simulate a collision.\n\n"
                "Inference requests, latencies, feature validation scores, and class distributions will "
                "be recorded in real time and persist across server restarts."
            )
        else:
            drift_cols = st.columns(2)
            with drift_cols[0]:
                st.markdown("##### Live Prediction Class Distribution vs Baseline")
                cur_dist = metrics["class_drift"].get("current_distribution", {})
                base_dist = metrics["class_drift"].get("baseline_distribution", {})

                dist_comparison = []
                for cls in ["Slight", "Serious", "Fatal"]:
                    dist_comparison.append({
                        "Severity Class": cls,
                        "Live Inferences (%)": f"{cur_dist.get(cls, 0.0) * 100:.1f}%",
                        "Training Baseline (%)": f"{base_dist.get(cls, 0.0) * 100:.1f}%",
                    })
                st.table(pd.DataFrame(dist_comparison))

            with drift_cols[1]:
                st.markdown("##### Operational Latency Telemetry")
                st.metric("P95 Latency", f"{metrics['p95_latency_ms']:.1f} ms" if metrics['p95_latency_ms'] is not None else "N/A")
                st.metric("Min / Max Latency", f"{metrics['min_latency_ms']:.1f} ms / {metrics['max_latency_ms']:.1f} ms" if metrics['min_latency_ms'] is not None else "N/A")

            st.markdown("##### Recent Live Predictions (Last 10 Records)")
            recent_logs = monitor.get_recent_logs(limit=10)
            if recent_logs:
                display_logs = []
                for log in recent_logs:
                    display_logs.append({
                        "ID": log["id"],
                        "Timestamp": log["created_at"],
                        "Predicted Severity": log["predicted_severity"],
                        "Confidence": f"{log['confidence'] * 100:.1f}%",
                        "Risk Alert Level": log["risk_alert_level"],
                        "Latency (ms)": f"{log['latency_ms']:.1f}",
                        "Speed Limit": f"{log['speed_limit']} mph" if log['speed_limit'] != "-" else "-",
                        "Weather": log["weather_condition"],
                        "Day of Week": log["day_of_week"],
                        "Status": log["status"],
                    })
                st.dataframe(pd.DataFrame(display_logs), use_container_width=True)

        st.info("Automated Retraining: When drift exceeds threshold (PSI >= 0.20), the scheduled Airflow DAG (`dags/mlops_pipeline_dag.py`) automatically retrains candidate models and registers the new champion version.")
