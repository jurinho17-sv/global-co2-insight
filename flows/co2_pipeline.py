"""Prefect 3 orchestration for the climate ML platform's Medallion data pipeline.

DAG:
    bronze_owid    \
    bronze_wdi      ->  validate_bronze  ->  transform_silver  ->  validate_silver
    bronze_paris   /                                                        |
                                                                            v
                                                              build_gold_dbt -> test_gold_dbt
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
from pathlib import Path

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from co2_ml.pipelines.ingest import download_owid_data
from co2_ml.pipelines.ingest_paris import build_ratification_table
from co2_ml.pipelines.ingest_worldbank import fetch_worldbank_wdi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse" / "co2_warehouse"


# =============================================================================
# Bronze
# =============================================================================
@task(retries=3, retry_delay_seconds=60, log_prints=True)
def ingest_bronze_owid(
    url: str = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
) -> str:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "owid_co2" / f"ingestion_date={today}"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(bronze_dir / "part-0.parquet")
    download_owid_data(url, output_path)
    create_markdown_artifact(key="bronze-owid", markdown=f"# Bronze OWID\n\nIngested from {url} -> `{output_path}`")
    return output_path


@task(retries=3, retry_delay_seconds=60, log_prints=True)
def ingest_bronze_worldbank() -> str:
    out = fetch_worldbank_wdi(PROJECT_ROOT / "data" / "bronze")
    create_markdown_artifact(key="bronze-worldbank", markdown=f"# Bronze WorldBank WDI\n\n-> `{out}`")
    return str(out)


@task(retries=2, log_prints=True)
def ingest_bronze_paris() -> str:
    out = build_ratification_table(PROJECT_ROOT / "data" / "bronze")
    create_markdown_artifact(key="bronze-paris", markdown=f"# Bronze Paris ratifications\n\n-> `{out}`")
    return str(out)


# =============================================================================
# GE validation gates
# =============================================================================
@task(log_prints=True)
def validate_bronze(bronze_owid_path: str) -> None:
    """Run the raw_owid GE suite against the Bronze OWID parquet. Raises on failure."""
    from tests.data.ge_validation import validate_bronze_owid_parquet

    if not validate_bronze_owid_parquet(bronze_owid_path):
        raise RuntimeError(f"Bronze GE validation FAILED for {bronze_owid_path}")
    print(f"[GE] Bronze validation PASSED for {bronze_owid_path}")


@task(log_prints=True)
def validate_silver(silver_parquet_path: str) -> None:
    """Run the silver_conformed GE suite against Silver output. Raises on failure."""
    from tests.data.ge_validation import validate_silver_conformed

    if not validate_silver_conformed(silver_parquet_path):
        raise RuntimeError(f"Silver GE validation FAILED for {silver_parquet_path}")
    print(f"[GE] Silver validation PASSED for {silver_parquet_path}")


# =============================================================================
# Silver — Spark on DataHub, pandas fallback locally
# =============================================================================
@task(log_prints=True)
def transform_silver(_bronze_owid_path: str) -> str:
    """Bronze -> Silver/conformed. Uses PySpark if available (DataHub); pandas fallback locally."""
    if importlib.util.find_spec("pyspark") is not None:
        print("[silver] pyspark detected — running silver_clean_spark.py via subprocess")
        result = subprocess.run(
            ["python", "-m", "co2_ml.pipelines.silver_clean_spark"],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("silver_clean_spark failed")
    else:
        print("[silver] pyspark not installed — running silver_clean_pandas.py (same schema)")
        from co2_ml.pipelines.silver_clean_pandas import silver_clean

        silver_clean(PROJECT_ROOT)

    return str(PROJECT_ROOT / "data" / "silver" / "conformed" / "country_year_panel.parquet")


# =============================================================================
# Gold — dbt build + dbt test
# =============================================================================
@task(log_prints=True)
def build_gold_dbt() -> None:
    print(f"[dbt] running models in {WAREHOUSE_DIR}")
    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", str(PROJECT_ROOT)],
        cwd=str(WAREHOUSE_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("dbt run failed")


@task(log_prints=True)
def test_gold_dbt() -> None:
    print(f"[dbt] running tests in {WAREHOUSE_DIR}")
    result = subprocess.run(
        ["dbt", "test", "--profiles-dir", str(PROJECT_ROOT)],
        cwd=str(WAREHOUSE_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("dbt test failed")


# =============================================================================
# Orchestrating flow
# =============================================================================
@flow(name="climate-pipeline", log_prints=True)
def climate_pipeline(
    owid_url: str = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
) -> dict:
    """Full Bronze->Silver->Gold pipeline with GE gates and dbt build/test."""

    owid_future = ingest_bronze_owid.submit(owid_url)
    wdi_future = ingest_bronze_worldbank.submit()
    paris_future = ingest_bronze_paris.submit()

    bronze_owid_path = owid_future.result()
    wdi_future.result()
    paris_future.result()

    validate_bronze(bronze_owid_path)
    silver_path = transform_silver(bronze_owid_path)
    validate_silver(silver_path)
    build_gold_dbt()
    test_gold_dbt()

    return {"status": "success", "silver_path": silver_path}


# Backward-compatible alias for the previous flow name.
co2_pipeline = climate_pipeline


if __name__ == "__main__":
    climate_pipeline.serve(
        name="climate-pipeline-dev",
        cron="0 6 * * *",
    )
