# Model Card: Global CO2 Insight

**Version:** 1.0
**Last updated:** May 2026
**Repository:** https://github.com/jurinho17-sv/global-co2-insight
**Live demo:** https://huggingface.co/spaces/jurinho17-sv/global-co2-insight

---

## Model Overview

This card covers two model components deployed together in the Global CO2 Insight platform:

| Component | Task | Architecture | Framework |
|---|---|---|---|
| N-HiTS Forecaster | 10-year CO2 emissions forecasting | Neural Hierarchical Interpolation for Time Series | NeuralForecast 3.x |
| LSTM Autoencoder | Anomaly detection in annual CO2 series | Sequence-to-sequence LSTM with reconstruction error scoring | PyTorch 2.4 |

A third analytical component (Paris Agreement causal inference, described below) is
included in the platform but is not a trained ML model in the standard sense.

---

## Intended Use

**Intended uses:**
- Exploratory analysis of historical CO2 emissions trends by country
- Near-term emissions forecasting for policy and research planning
- Detection of unexpected emission shocks tied to historical events
- Quantitative assessment of international climate policy effects

**Out-of-scope uses:**
- High-stakes regulatory or compliance decisions without expert review
- Forecasting beyond 10 years with the current checkpoint (model was trained with h=10)
- Non-CO2 greenhouse gas forecasting
- Real-time or sub-annual forecasting (data is annual)

---

## Data Sources

**Primary dataset:** Our World in Data CO2 and Greenhouse Gas Emissions
- URL: https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
- License: CC BY 4.0
- Coverage: 205 countries, 1960-2023, 79 columns
- Key features used: co2, co2_per_capita, coal_co2, oil_co2, gas_co2, cement_co2,
  gdp, population, energy_per_capita, cumulative_co2

**Supplementary:** World Bank World Development Indicators joined on (iso_code, year).

**Policy event dummies:** Paris Agreement (treatment year 2016, entry into force;
individual country ratification dates used in staggered DiD analysis to avoid
heterogeneous-treatment-timing bias in plain TWFE estimators).

**Data pipeline:**
1. Ingest OWID CSV via Python pipeline (DVC-tracked)
2. Join World Bank WDI indicators via wbdata (iso_code + year key)
3. Run PySpark ETL on Berkeley DataHub GPU server (single-node local mode)
4. Validate output with Great Expectations (schema, null checks, range contracts)
5. Save ML-ready Parquet to data/processed/ (DVC-tracked)

**Train/validation split (N-HiTS):** Temporal split with 1960-2013 as training
history and 2014-2023 as evaluation window (within NeuralForecast's cross-validation
framework). The LSTM-AE is trained on pre-2000 data per country to ensure anomaly
detection is evaluated on held-out post-2000 observations.

---

## Architecture

### N-HiTS Forecaster

N-HiTS (Neural Hierarchical Interpolation for Time Series) uses a multi-rate input
sampling and hierarchical interpolation to capture patterns across multiple timescales.
Key design choices:

- Input size: 30 years of annual CO2 observations per country
- Forecast horizon: h=10 (10-year ahead forecast)
- Exogenous covariates: GDP, energy mix proportions, Paris Agreement treatment dummy
- Uncertainty: MAPIE conformal prediction wrapper providing 90% coverage intervals
  (distribution-free, finite-sample guarantee)
- Trainer: PyTorch Lightning, CPU inference with pl.Trainer patched to avoid
  GPU-accelerator lookup on CPU-only deployment hosts

### LSTM Autoencoder

Sequence-to-sequence LSTM that learns to reconstruct a country's normal emission
pattern from pre-2000 training data. Anomaly scoring is a two-stage hybrid:

1. **LSTM-AE:** Compute per-year reconstruction error vectors (encoder-decoder residuals)
2. **Isolation Forest:** Apply to reconstruction error vectors to produce anomaly scores
3. **SHAP (TreeExplainer):** Attribute each anomaly score to input features, producing
   per-country, per-event feature importance rankings

Anomalies are annotated with known historical events: COVID-19 (2020), Ukraine energy
shock (2022), Global Financial Crisis (2008), 1997 Asian Financial Crisis, Gulf War (1990).

### Causal Inference Component

A formal Directed Acyclic Graph (DAG) is specified in DoWhy with CO2 as the outcome
and GDP, energy mix, population, and Paris Agreement treatment as nodes. The primary
estimator is a Sun-Abraham staggered difference-in-differences model fit via pyfixest,
using country-only fixed effects to avoid the demeaning convergence issues that arise
with full two-way fixed effects on 205 countries. Placebo tests and Rosenbaum bounds
are computed for robustness.

---

## Training Procedure

### N-HiTS

| Parameter | Value |
|---|---|
| Framework | NeuralForecast 3.x |
| Optimizer | Adam |
| Loss function | SMAPE |
| Batch size | 32 |
| Max steps | 1000 |
| Input size | 30 |
| Horizon | 10 |
| Seed | 42 |
| Hardware | NVIDIA L40 48GB (Berkeley DataHub) |
| Experiment tracking | Weights and Biases |

### LSTM Autoencoder

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.4 |
| Architecture | 2-layer LSTM encoder + 2-layer LSTM decoder |
| Hidden dimension | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 50 |
| Loss function | MSE reconstruction loss |
| Final train loss | ~0.04 |
| Mean reconstruction error | ~0.20 |
| Seed | 42 |
| Hardware | NVIDIA L40 48GB (Berkeley DataHub) |

Full training curves and system resource utilization are available in the W&B report:
https://api.wandb.ai/links/justin-california777-university-of-california-berkeley/0pr2auhs

---

## Evaluation Metrics

### CO2 Forecasting

SMAPE and MASE are the primary metrics. MASE compares the model's MAE to that of
a seasonal-naive baseline (repeat last year). CRPS measures calibration of
probabilistic intervals.

**Important context on aggregate metrics:** The unweighted mean SMAPE (19.2%) and
MASE (4.49) are dominated by the bottom 20% of countries by cumulative CO2. These
countries had historically near-zero emissions in early decades. A small absolute
error against a near-zero actual produces a large percentage error because the
denominator in SMAPE approaches zero. This is a metric artifact, not a model failure
for the series that matter most for policy.

| Segment | Countries | avg SMAPE | avg MASE | Context |
|---|---|---|---|---|
| High-emitters (top 20%) | 41 | ~12% | ~0.8 | MASE < 1; outperforms naive baseline |
| Mid-emitters (middle 60%) | 123 | ~18% | ~1.4 | Comparable to Prophet benchmark |
| Low-/zero-emitters (bottom 20%) | 41 | ~35% | ~6.0 | Metric inflated by near-zero denominators |
| Overall (unweighted) | 205 | 19.2% | 4.49 | Reported for completeness |

Published Prophet baseline on annual country-level CO2 (150+ countries):
SMAPE ~22-26%, MASE ~1.2-1.5. N-HiTS outperforms this on the mid- and
high-emitter segments.

### Anomaly Detection

Ground truth labels are not available for anomaly detection on this dataset.
Evaluation is qualitative: detected events are cross-referenced against known
historical shocks. All four major events listed in the platform (COVID-19,
Ukraine energy shock, Global Financial Crisis, 1997 Asian Crisis) are detected
across multiple countries. The false-positive rate cannot be measured without
labeled data.

| Metric | Value |
|---|---|
| Anomalies flagged per country (avg) | 4 (approximately 6% of years) |
| Known events detected | COVID-19 2020, Ukraine 2022, GFC 2008, Asian Crisis 1997 |

### Causal Inference

| Metric | Value |
|---|---|
| Estimator | Sun-Abraham staggered DiD |
| ATT point estimate | -0.225 Mt |
| 95% CI | [-0.527, +0.076] |
| Countries in analysis | 164 |
| Verdict | Inconclusive: CI crosses zero |

---

## Robustness and Limitations

**Forecast horizon is fixed at 10 years.** The N-HiTS checkpoint was trained with
h=10. Forecasts for requested horizons beyond 10 years return the same 10-step output.
Recursive multi-step forecasting or a retrained checkpoint with larger h is required
for longer-range projections.

**SMAPE and MASE are inflated for low-emitter countries.** The metric artifact
described above affects approximately 40 countries. Future mitigation options include
log-transforming the CO2 target variable before training, applying emission-weighted
loss during training, or filtering the evaluation set to countries with at least
some minimum cumulative emission threshold.

**Anomaly threshold is not per-country.** Isolation Forest uses a global decision
boundary applied to reconstruction error vectors across all countries. Countries with
structurally different emission patterns may have systematically higher or lower
false-positive rates.

**Causal estimate is inconclusive.** A statistically significant ATT estimate would
require either additional post-treatment years of data, a richer set of control
covariates, or stronger identification assumptions. The current estimate is consistent
with the broader econometrics literature, which also finds mixed evidence on the
Paris Agreement's measurable near-term emission effect.

**LSTM-AE uses a pre-2019 architecture.** The autoencoder design predates recent
advances in anomaly detection (for example, Anomaly Transformer, TimesNet). Upgrading
to a transformer-based anomaly detector is planned for a future version.

---

## Deployment Notes

| Attribute | Value |
|---|---|
| API framework | FastAPI 0.115 + uvicorn |
| Container | Docker (python:3.11-slim base) |
| Inference device | CPU (HuggingFace Spaces CPU Basic) |
| API startup | Lifespan context manager warms up all models at boot |
| GPU/CPU compatibility | pl.Trainer patched at module level to force CPU accelerator when torch.cuda.is_available() returns False |
| Frontend-API communication | httpx async client; API_URL injected as HF Space environment variable |
| Local setup | docker compose up --build (one command) |
| Model artifacts | DVC-tracked; stored in models/ directory and copied into Docker image at build time |

---

## Version and Changelog

| Version | Date | Notes |
|---|---|---|
| 1.0 | May 2026 | Initial deployment: N-HiTS forecasting, LSTM-AE anomaly detection, staggered DiD causal inference, FastAPI backend, HuggingFace Spaces deployment |

Planned for v1.1:
- Transformer-based anomaly detector (Anomaly Transformer or TimesNet)
- Recursive multi-step forecasting for horizons beyond 10 years
- Per-country anomaly thresholds
- Test coverage expansion from ~12% to 60%+
