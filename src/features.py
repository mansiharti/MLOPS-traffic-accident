import json
import os
import numpy as np
import pandas as pd
TARGET_COLUMN = "accident_severity"
DATE_COLUMN = "accident_date"

SEVERITY_CLASSES = ["Fatal", "Serious", "Slight"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SEVERITY_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(SEVERITY_CLASSES)}

NUMERICAL_FEATURES = [
    "accident_hour",
    "speed_limit",
    "number_of_vehicles",
    "number_of_casualties",
    "temperature",
    "precipitation",
    "wind_speed",
]

CATEGORICAL_FEATURES = [
    "road_type",
    "light_conditions",
    "road_surface_conditions",
    "weather_condition",
    "day_of_week",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """
    Constructs an end-to-end ColumnTransformer for tabular collision telemetry.
    Strict Featurization Ordering: Failsafe imputers and encoders fitted exclusively on train split.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on accident_date.
    Avoids future lookahead bias in model evaluation.
    """
    df_sorted = df.copy()
    df_sorted[DATE_COLUMN] = pd.to_datetime(df_sorted[DATE_COLUMN])
    df_sorted = df_sorted.sort_values(by=[DATE_COLUMN, "accident_time"]).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"[Chronological Split] Total: {n:,} rows")
    print(f"  - Train : {len(train_df):,} rows ({train_df[DATE_COLUMN].min().date()} to {train_df[DATE_COLUMN].max().date()})")
    print(f"  - Val   : {len(val_df):,} rows ({val_df[DATE_COLUMN].min().date()} to {val_df[DATE_COLUMN].max().date()})")
    print(f"  - Test  : {len(test_df):,} rows ({test_df[DATE_COLUMN].min().date()} to {test_df[DATE_COLUMN].max().date()})")

    return train_df, val_df, test_df


def extract_baseline_distribution(train_df: pd.DataFrame) -> dict:
    """
    Extracts numerical quantiles and categorical frequency distributions
    to serve as reference baseline for drift detection.
    """
    baseline = {
        "record_count": len(train_df),
        "target_distribution": train_df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
        "numerical_stats": {},
        "categorical_distribution": {},
    }

    for col in NUMERICAL_FEATURES:
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        baseline["numerical_stats"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "quantiles": [float(series.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
        }

    for col in CATEGORICAL_FEATURES:
        counts = train_df[col].astype(str).value_counts(normalize=True).to_dict()
        baseline["categorical_distribution"][col] = {k: float(v) for k, v in counts.items()}

    return baseline

