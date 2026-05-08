"""Great Expectations 1.x Fluent-API validation suites for raw and processed CO2 data."""

from __future__ import annotations

import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_context() -> gx.data_context.AbstractDataContext:
    return gx.get_context(mode="file", project_root_dir=str(REPO_ROOT))


def _print_results(label: str, success: bool, results: list) -> None:
    print(f"[GE] {label} validation: {'PASS' if success else 'FAIL'}")
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {r.expectation_config.type}")


def validate_raw_owid(csv_path: str) -> bool:
    context = _get_context()

    data_source = context.data_sources.add_or_update_pandas(name="raw_csv_source")
    data_asset = data_source.add_dataframe_asset(name="raw_csv_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("raw_csv_batch")

    suite = context.suites.add_or_update(gx.core.ExpectationSuite(name="raw_owid_suite"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="country"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="year"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="iso_code"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="co2"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="country"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="year"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="year", min_value=1750, max_value=2030))
    # GE 1.15.1 has no ExpectTableRowCountToBeGreaterThan; "> 10000" == "min_value=10001".
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=10001))

    df = pd.read_csv(csv_path)

    val_def = context.validation_definitions.add_or_update(
        gx.core.ValidationDefinition(
            name="raw_owid_validation",
            data=batch_def,
            suite=suite,
        )
    )
    result = val_def.run(batch_parameters={"dataframe": df})
    _print_results("Raw OWID", result.success, result.results)
    return bool(result.success)


def validate_silver_conformed(parquet_path: str) -> bool:
    """Run silver_conformed_suite against a 25-column conformed Silver parquet."""
    context = _get_context()

    data_source = context.data_sources.add_or_update_pandas(name="silver_conformed_source")
    data_asset = data_source.add_dataframe_asset(name="silver_conformed_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("silver_conformed_batch")

    suite = context.suites.add_or_update(gx.core.ExpectationSuite(name="silver_conformed_suite"))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=10000, max_value=20000))
    suite.add_expectation(gx.expectations.ExpectTableColumnCountToEqual(value=25))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="iso_code"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="year"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="year", min_value=1960, max_value=2024))
    # OWID core columns retained
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="co2"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="country"))
    # Joined fields from World Bank WDI
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="gdp_growth_pct"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="urban_population_pct"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="manufacturing_pct_gdp"))
    # Joined + derived fields from Paris ratifications
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="ratification_year"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="paris_treated"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="years_since_ratification"))

    df = pd.read_parquet(parquet_path)

    unique_countries = df["iso_code"].nunique()
    if unique_countries < 150:
        print(f"[GE] WARNING: Only {unique_countries} unique countries (expected >= 150)")

    val_def = context.validation_definitions.add_or_update(
        gx.core.ValidationDefinition(
            name="silver_conformed_validation",
            data=batch_def,
            suite=suite,
        )
    )
    result = val_def.run(batch_parameters={"dataframe": df})
    _print_results("Silver conformed", result.success, result.results)
    return bool(result.success) and (unique_countries >= 150)


# Backward-compatible alias used by older callers; routes to the new conformed suite.
validate_processed_parquet = validate_silver_conformed


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_ok = validate_raw_owid(str(repo_root / "data" / "raw" / "owid-co2-data.csv"))
    silver_ok = validate_silver_conformed(
        str(repo_root / "data" / "silver" / "conformed" / "country_year_panel.parquet")
    )
    print(f"\n[GE] Overall: {'ALL PASS' if (raw_ok and silver_ok) else 'FAILURES DETECTED'}")
    sys.exit(0 if (raw_ok and silver_ok) else 1)
