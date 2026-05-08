"""Bronze ingestion: World Bank WDI indicators via wbdata."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import wbdata

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "NV.IND.MANF.ZS": "manufacturing_pct_gdp",
}

DATE_RANGE = (dt.datetime(1960, 1, 1), dt.datetime(2023, 12, 31))


def fetch_worldbank_wdi(output_root: str | Path) -> Path:
    """Fetch WDI indicators for all available countries (1960-2023) and write Bronze parquet."""
    df = wbdata.get_dataframe(INDICATORS, date=DATE_RANGE)
    df = df.reset_index().rename(columns={"date": "year", "country": "country_name"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])

    iso_lookup = {c["name"]: c["id"] for c in wbdata.get_countries()}
    df["iso_code"] = df["country_name"].map(iso_lookup)
    df = df.dropna(subset=["iso_code"])

    now = dt.datetime.now(dt.timezone.utc)
    df["_ingested_at"] = now
    df["_source"] = "worldbank_wdi"

    today = now.date().isoformat()
    out_dir = Path(output_root) / "worldbank_wdi" / f"ingestion_date={today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part-0.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows × {len(df.columns)} cols to {out_path}")
    print(f"Indicators: {list(INDICATORS.values())}")
    print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")
    print(f"Countries: {df['iso_code'].nunique()}")
    return out_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    fetch_worldbank_wdi(project_root / "data" / "bronze")


if __name__ == "__main__":
    main()
