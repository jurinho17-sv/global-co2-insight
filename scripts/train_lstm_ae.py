"""LSTM Autoencoder training for anomaly detection.

Usage:
    python scripts/train_lstm_ae.py

Reads config from configs/model/lstm_ae.yaml, trains on pre-2000 data from
data/gold/ml_features.parquet, logs metrics to W&B project "global-co2-insight",
and saves the model to models/lstm_ae/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIGS_DIR = PROJECT_ROOT / "configs" / "model"
DATA_PATH = PROJECT_ROOT / "data" / "gold" / "ml_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "lstm_ae"

FEATURES = ["co2", "gdp", "primary_energy_consumption", "population"]
WINDOW_SIZE = 10


def _load_config() -> dict:
    with open(CONFIGS_DIR / "lstm_ae.yaml") as f:
        return yaml.safe_load(f)


def build_sequences(
    df: pd.DataFrame,
    features: list[str],
    window: int,
) -> np.ndarray:
    """Build sliding-window sequences per country.

    Returns array of shape (n_samples, window, n_features).
    """
    sequences = []
    for _, group in df.groupby("iso_code"):
        vals = group[features].values
        for i in range(len(vals) - window + 1):
            sequences.append(vals[i : i + window])
    return (
        np.array(sequences, dtype=np.float32) if sequences else np.empty((0, window, len(features)), dtype=np.float32)
    )


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature standardization across all samples.

    Returns (X_norm, means, stds).
    """
    # Reshape to (n_samples * window, n_features) for stats
    orig_shape = X.shape
    flat = X.reshape(-1, orig_shape[-1])
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    X_norm = (X - means) / stds
    return X_norm, means, stds


def main() -> None:
    import wandb

    from co2_ml.models.anomaly import LSTMAutoencoder, compute_reconstruction_errors

    cfg = _load_config()
    hidden_dim = cfg.get("hidden_dim", 64)
    num_layers = cfg.get("num_layers", 2)
    epochs = cfg.get("epochs", 50)
    batch_size = cfg.get("batch_size", 32)
    lr = float(cfg.get("lr", 1e-3))
    train_cutoff_year = cfg.get("train_cutoff_year", 2000)
    seed = cfg.get("seed", 42)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["iso_code"])

    # Fill missing feature values per country
    available_features = [f for f in FEATURES if f in df.columns]
    for feat in available_features:
        df[feat] = df.groupby("iso_code")[feat].transform(lambda s: s.ffill().fillna(0))

    # Train on pre-cutoff data
    train_df = df[df["year"] < train_cutoff_year].copy()
    full_df = df.copy()

    print(f"Train cutoff year: {train_cutoff_year}")
    print(f"Train data: {len(train_df)} rows, Full data: {len(full_df)} rows")

    # Build sequences
    X_train = build_sequences(train_df, available_features, WINDOW_SIZE)
    X_full = build_sequences(full_df, available_features, WINDOW_SIZE)

    if len(X_train) == 0:
        print("ERROR: No training sequences could be built. Check data.")
        return

    print(f"Train sequences: {X_train.shape}, Full sequences: {X_full.shape}")

    # Normalize
    X_train_norm, means, stds = normalize(X_train)
    X_full_norm = (X_full - means) / stds

    # --- W&B ---
    wandb.init(
        project="global-co2-insight",
        name="lstm-ae-train",
        config={
            "model": "lstm-ae",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_cutoff_year": train_cutoff_year,
            "seed": seed,
            "n_features": len(available_features),
            "window_size": WINDOW_SIZE,
            "n_train_sequences": len(X_train_norm),
        },
    )

    # --- Model ---
    n_features = len(available_features)
    model = LSTMAutoencoder(input_dim=n_features, hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train_norm))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Training loop ---
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs} — Loss: {avg_loss:.6f}")

    # --- Reconstruction errors on full dataset ---
    model = model.cpu()
    errors = compute_reconstruction_errors(model, X_full_norm)

    wandb.log(
        {
            "final_train_loss": avg_loss,
            "recon_error_mean": float(errors.mean()),
            "recon_error_std": float(errors.std()),
            "recon_error_max": float(errors.max()),
            "recon_error_p95": float(np.percentile(errors, 95)),
            "recon_error_p99": float(np.percentile(errors, 99)),
        }
    )

    print(
        f"\nReconstruction errors — Mean: {errors.mean():.6f}, "
        f"Std: {errors.std():.6f}, P95: {np.percentile(errors, 95):.6f}"
    )

    # --- Save ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lstm_ae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "means": means,
            "stds": stds,
            "config": cfg,
            "features": available_features,
            "window_size": WINDOW_SIZE,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
