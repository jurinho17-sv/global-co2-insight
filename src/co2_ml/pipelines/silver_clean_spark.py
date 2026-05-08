"""Silver cleansing+conforming via PySpark — DataHub-only execution.

Reads Bronze partitioned parquet (OWID + WorldBank WDI + Paris ratifications),
joins on (iso_code, year), cleanses, deduplicates, derives treatment columns,
and writes a single Silver `country_year_panel.parquet`.

Run on DataHub:
    spark-submit --master local[*] src/co2_ml/pipelines/silver_clean_spark.py
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("co2-silver-clean")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def silver_clean(project_root: Path) -> Path:
    spark = build_spark()

    bronze_root = project_root / "data" / "bronze"
    silver_dir = project_root / "data" / "silver" / "conformed"
    silver_dir.mkdir(parents=True, exist_ok=True)
    silver_path = silver_dir / "country_year_panel.parquet"

    owid = (
        spark.read.parquet(str(bronze_root / "owid_co2"))
        .filter(F.col("iso_code").isNotNull())
        .filter(F.col("year") >= 1960)
        .filter((F.col("co2").isNull()) | (F.col("co2") >= 0))
    )

    wdi = spark.read.parquet(str(bronze_root / "worldbank_wdi"))
    wdi = wdi.select(
        "iso_code",
        "year",
        "gdp_growth_pct",
        "urban_population_pct",
        "manufacturing_pct_gdp",
    )

    paris = spark.read.parquet(str(bronze_root / "paris_ratifications"))
    paris = paris.select("iso_code", "ratification_year")

    panel = (
        owid.join(wdi, on=["iso_code", "year"], how="left")
        .join(paris, on=["iso_code"], how="left")
        .dropDuplicates(["iso_code", "year"])
        .withColumn(
            "paris_treated",
            F.when(
                F.col("ratification_year").isNotNull() & (F.col("year") >= F.col("ratification_year")),
                F.lit(True),
            ).otherwise(F.lit(False)),
        )
        .withColumn(
            "years_since_ratification",
            F.when(
                F.col("ratification_year").isNotNull(),
                F.greatest(F.lit(0), F.col("year") - F.col("ratification_year")),
            ).otherwise(F.lit(0)),
        )
    )

    panel.write.mode("overwrite").parquet(str(silver_path))

    row_count = panel.count()
    print(f"Wrote Silver country_year_panel: {row_count:,} rows -> {silver_path}")
    print("Schema:")
    panel.printSchema()

    spark.stop()
    return silver_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    silver_clean(project_root)


if __name__ == "__main__":
    main()
