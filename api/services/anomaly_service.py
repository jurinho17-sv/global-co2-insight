"""Anomaly detection service — LSTM-AE + Isolation Forest pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.schemas.anomaly import AnomalyRecord, AnomalyResponse

FEATURE_COLS = ["co2", "gdp", "primary_energy_consumption", "population"]


def detect_anomalies(
    model: Any,
    norm_stats: dict,
    df: pd.DataFrame,
    country_iso: str,
) -> AnomalyResponse:
    from co2_ml.models.anomaly import (
        compute_reconstruction_errors,
        label_known_events,
        run_isolation_forest,
    )

    country_df = df[df["iso_code"] == country_iso].sort_values("year").copy()
    # Gold parquet has population as Int64 (nullable) and other features as float64.
    # .values on the mixed-dtype frame collapses to object dtype, which torch.tensor rejects.
    # Coerce to float64 first so the LSTM-AE input tensor builds cleanly.
    features = country_df[FEATURE_COLS].fillna(0).astype(float).values

    # Z-score normalize using training statistics
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    std_safe = np.where(std == 0, 1.0, std)
    normalized = (features - mean) / std_safe

    # Reshape for LSTM-AE: (n_samples, seq_len=1, n_features)
    X = normalized.reshape(-1, 1, len(FEATURE_COLS))

    errors = compute_reconstruction_errors(model, X)
    labels = run_isolation_forest(errors, contamination=0.05)

    anomaly_df = pd.DataFrame(
        {
            "year": country_df["year"].values,
            "is_anomaly": labels == -1,
            "reconstruction_error": errors,
        }
    )
    anomaly_df = label_known_events(anomaly_df)

    records = [
        AnomalyRecord(
            year=int(row["year"]),
            is_anomaly=bool(row["is_anomaly"]),
            reconstruction_error=float(row["reconstruction_error"]),
            event_label=row["event_label"],
        )
        for _, row in anomaly_df.iterrows()
    ]

    return AnomalyResponse(
        country=country_iso,
        anomalies=records,
        total_anomalies=sum(1 for r in records if r.is_anomaly),
    )
