"""Bronze ingestion: OWID CO2 dataset.

Writes a partitioned, append-only Parquet snapshot under
data/bronze/owid_co2/ingestion_date=YYYY-MM-DD/part-0.parquet
with standardized provenance columns (_ingested_at, _source_url).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml


def download_owid_data(url: str, output_path: str) -> None:
    """Download OWID CO2 CSV from URL and persist as Bronze parquet (with provenance).

    `output_path` may be a `.parquet` file or a directory; either way the file
    is written with the standard provenance columns. CSV destinations are still
    accepted for backward compatibility but Bronze should be parquet.
    """
    print(f"Downloading OWID CO2 data from: {url}")
    df = pd.read_csv(url)

    df["_ingested_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    df["_source_url"] = url

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Year range: {year_min}-{year_max}")
    print(f"  Saved to: {out}")


def main() -> None:
    params_path = Path(__file__).resolve().parent.parent.parent.parent / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    project_root = params_path.parent
    url = params["data"]["owid_url"]

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    bronze_dir = project_root / "data" / "bronze" / "owid_co2" / f"ingestion_date={today}"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(bronze_dir / "part-0.parquet")

    download_owid_data(url, output_path)


if __name__ == "__main__":
    main()
