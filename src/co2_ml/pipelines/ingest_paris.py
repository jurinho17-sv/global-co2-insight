"""Bronze ingestion: per-country Paris Agreement ratification dates.

Source: UNFCCC ratification register (https://unfccc.int/process/the-paris-agreement/status-of-ratification).
Curated subset of major emitters; expand as needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARIS_SOURCE_URL = "https://unfccc.int/process/the-paris-agreement/status-of-ratification"

# (iso_code, country_name, ratification_year)
RATIFICATIONS: list[tuple[str, str, int]] = [
    ("USA", "United States", 2016),
    ("CHN", "China", 2016),
    ("IND", "India", 2016),
    ("RUS", "Russia", 2019),
    ("JPN", "Japan", 2016),
    ("DEU", "Germany", 2016),
    ("KOR", "South Korea", 2016),
    ("CAN", "Canada", 2016),
    ("GBR", "United Kingdom", 2016),
    ("FRA", "France", 2016),
    ("ITA", "Italy", 2016),
    ("AUS", "Australia", 2016),
    ("BRA", "Brazil", 2016),
    ("ZAF", "South Africa", 2016),
    ("MEX", "Mexico", 2016),
    ("IDN", "Indonesia", 2016),
    ("SAU", "Saudi Arabia", 2016),
    ("TUR", "Turkey", 2016),
    ("ARG", "Argentina", 2016),
    ("POL", "Poland", 2016),
    ("NLD", "Netherlands", 2016),
    ("BEL", "Belgium", 2016),
    ("ESP", "Spain", 2016),
    ("SWE", "Sweden", 2016),
    ("NOR", "Norway", 2016),
    ("CHE", "Switzerland", 2016),
    ("NZL", "New Zealand", 2016),
    ("PRT", "Portugal", 2016),
    ("GRC", "Greece", 2016),
    ("CZE", "Czechia", 2017),
    ("HUN", "Hungary", 2016),
    ("ROU", "Romania", 2016),
    ("BGD", "Bangladesh", 2016),
    ("PAK", "Pakistan", 2016),
    ("NGA", "Nigeria", 2016),
    ("ETH", "Ethiopia", 2016),
    ("EGY", "Egypt", 2017),
    ("IRN", "Iran", 2017),
    ("IRQ", "Iraq", 2021),
    ("DZA", "Algeria", 2016),
]


def build_ratification_table(output_root: str | Path) -> Path:
    df = pd.DataFrame(RATIFICATIONS, columns=["iso_code", "country", "ratification_year"])
    df["_ingested_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    df["_source_url"] = PARIS_SOURCE_URL

    out_dir = Path(output_root) / "paris_ratifications"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ratification_dates.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df):,} ratification rows to {out_path}")
    print(f"Year range: {int(df['ratification_year'].min())}-{int(df['ratification_year'].max())}")
    return out_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    build_ratification_table(project_root / "data" / "bronze")


if __name__ == "__main__":
    main()
