"""Silver cleansing+conforming via pandas — local fallback when PySpark is absent.

Mirrors src/co2_ml/pipelines/silver_clean_spark.py exactly (same join keys,
same dedup, same derived columns, same final 25-column projection) so the
schemas/silver_country_year.yaml contract is satisfied regardless of which
runner produced the file. PySpark on DataHub is the production path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "iso_code",
    "country",
    "year",
    "co2",
    "co2_per_capita",
    "co2_growth_pct",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2",
    "flaring_co2",
    "total_ghg",
    "methane",
    "nitrous_oxide",
    "primary_energy_consumption",
    "population",
    "gdp",
    "gdp_growth_pct",
    "urban_population_pct",
    "manufacturing_pct_gdp",
    "ratification_year",
    "paris_treated",
    "years_since_ratification",
    "_ingested_at",
    "_source_url",
]


def silver_clean(project_root: Path) -> Path:
    bronze = project_root / "data" / "bronze"

    owid = pd.read_parquet(bronze / "owid_co2")
    owid = owid[owid["iso_code"].notna()]
    owid = owid[owid["year"] >= 1960]
    owid = owid[(owid["co2"].isna()) | (owid["co2"] >= 0)]
    owid["year"] = owid["year"].astype("int64")

    wdi_dir = bronze / "worldbank_wdi"
    wdi_part = sorted(wdi_dir.glob("ingestion_date=*/part-*.parquet"))[-1]
    wdi = pd.read_parquet(wdi_part)[
        ["iso_code", "year", "gdp_growth_pct", "urban_population_pct", "manufacturing_pct_gdp"]
    ]
    wdi["year"] = wdi["year"].astype("int64")

    paris = pd.read_parquet(bronze / "paris_ratifications" / "ratification_dates.parquet")[
        ["iso_code", "ratification_year"]
    ]

    panel = (
        owid.merge(wdi, on=["iso_code", "year"], how="left")
        .merge(paris, on=["iso_code"], how="left")
        .drop_duplicates(["iso_code", "year"])
    )
    panel["paris_treated"] = panel["ratification_year"].notna() & (panel["year"] >= panel["ratification_year"])
    panel["years_since_ratification"] = np.where(
        panel["ratification_year"].notna(),
        np.maximum(0, panel["year"] - panel["ratification_year"]),
        0,
    ).astype("int64")

    panel = panel.rename(columns={"co2_growth_prct": "co2_growth_pct"})
    panel = panel[CANONICAL_COLUMNS].copy()
    panel["year"] = panel["year"].astype("int64")
    panel["paris_treated"] = panel["paris_treated"].astype("bool")
    panel["ratification_year"] = panel["ratification_year"].astype("Int64")
    panel["years_since_ratification"] = panel["years_since_ratification"].astype("int64")

    out_dir = project_root / "data" / "silver" / "conformed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "country_year_panel.parquet"
    panel.to_parquet(out_path, index=False)

    treated_pct = panel["paris_treated"].mean() * 100
    print(f"Wrote Silver country_year_panel: {len(panel):,} rows × {len(panel.columns)} cols -> {out_path}")
    print(f"Countries: {panel['iso_code'].nunique()}; year range: {panel['year'].min()}-{panel['year'].max()}")
    print(f"paris_treated TRUE: {int(panel['paris_treated'].sum())} ({treated_pct:.2f}%)")
    return out_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    silver_clean(project_root)


if __name__ == "__main__":
    main()
