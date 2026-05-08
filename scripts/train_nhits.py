"""N-HiTS fine-tuning on all countries via NeuralForecast.

Usage:
    python scripts/train_nhits.py

Reads config from configs/model/nhits.yaml, trains on data/gold/ml_features.parquet,
logs metrics to W&B project "global-co2-insight", and saves the model to models/nhits/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIGS_DIR = PROJECT_ROOT / "configs" / "model"
DATA_PATH = PROJECT_ROOT / "data" / "gold" / "ml_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "nhits"


def _load_config() -> dict:
    with open(CONFIGS_DIR / "nhits.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    import wandb
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    from co2_ml.utils.metrics import mase, smape

    cfg = _load_config()
    horizon = cfg.get("horizon", 10)
    input_size = cfg.get("input_size", 20)
    max_steps = cfg.get("max_steps", 100)
    learning_rate = float(cfg.get("learning_rate", 2e-5))
    seed = cfg.get("seed", 42)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Data ---
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["co2", "iso_code"])

    # Build NeuralForecast format for all countries
    nf_df = pd.DataFrame(
        {
            "unique_id": df["iso_code"],
            "ds": pd.to_datetime(df["year"], format="%Y"),
            "y": df["co2"].values,
        }
    )

    covariates = ["gdp", "primary_energy_consumption", "population"]
    for cov in covariates:
        if cov in df.columns:
            nf_df[cov] = df[cov].ffill().fillna(0).values

    nf_df = nf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Filter countries with enough data points
    counts = nf_df.groupby("unique_id").size()
    valid_ids = counts[counts >= input_size + horizon].index
    nf_df = nf_df[nf_df["unique_id"].isin(valid_ids)].reset_index(drop=True)
    print(f"Training on {len(valid_ids)} countries with >= {input_size + horizon} data points")

    # --- W&B ---
    wandb.init(
        project="global-co2-insight",
        name="nhits-train",
        config={
            "model": "nhits",
            "horizon": horizon,
            "input_size": input_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "n_countries": len(valid_ids),
        },
    )

    # --- Model ---
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using accelerator: {accelerator}")

    models = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            random_seed=seed,
            accelerator=accelerator,
        )
    ]

    nf = NeuralForecast(models=models, freq="YS")

    # --- Train/Val split: hold out last `horizon` years per country ---
    val_dfs = []
    train_dfs = []
    for uid in valid_ids:
        uid_df = nf_df[nf_df["unique_id"] == uid]
        train_dfs.append(uid_df.iloc[:-horizon])
        val_dfs.append(uid_df.iloc[-horizon:])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # --- Fit ---
    nf.fit(df=train_df)
    forecast_df = nf.predict()

    # --- Evaluate ---
    print(f"\nForecast columns: {list(forecast_df.columns)}")
    print(f"Forecast head:\n{forecast_df.head()}")

    mase_scores = []
    smape_scores = []

    for uid in valid_ids:
        uid_val = val_df[val_df["unique_id"] == uid]["y"].values
        uid_fc = forecast_df[forecast_df["unique_id"] == uid]
        if uid_fc.empty or "NHITS" not in uid_fc.columns:
            continue
        uid_pred = uid_fc["NHITS"].values

        if len(uid_pred) != len(uid_val):
            continue

        m = mase(uid_val, uid_pred)
        s = smape(uid_val, uid_pred)
        if np.isfinite(m) and np.isfinite(s):
            mase_scores.append(m)
            smape_scores.append(s)
            wandb.log({"country": uid, "mase": m, "smape": s})

    avg_mase = float(np.mean(mase_scores)) if mase_scores else float("nan")
    avg_smape = float(np.mean(smape_scores)) if smape_scores else float("nan")

    wandb.log({"avg_mase": avg_mase, "avg_smape": avg_smape})
    print(f"\nResults — Avg MASE: {avg_mase:.4f}, Avg sMAPE: {avg_smape:.4f}")

    # --- Save ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nf.save(str(MODEL_DIR), overwrite=True)
    print(f"Model saved to {MODEL_DIR}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
