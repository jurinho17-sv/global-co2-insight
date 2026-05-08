{{ config(materialized='table') }}

WITH base AS (
    SELECT *
    FROM {{ source('silver_conformed', 'country_year_panel') }}
)

SELECT
    iso_code,
    country,
    CAST(year AS INTEGER)                                              AS year,
    CAST(co2 AS DOUBLE)                                                AS co2,
    CAST(co2_per_capita AS DOUBLE)                                     AS co2_per_capita,
    CAST(gdp AS DOUBLE)                                                AS gdp,
    CAST(population AS BIGINT)                                         AS population,
    CAST(primary_energy_consumption AS DOUBLE)                         AS primary_energy_consumption,
    CAST(gdp_growth_pct AS DOUBLE)                                     AS gdp_growth_rate,
    -- Energy-mix shares as fractions of total CO2
    CASE WHEN co2 > 0 THEN coal_co2   / co2 END                        AS coal_pct,
    CASE WHEN co2 > 0 THEN oil_co2    / co2 END                        AS oil_pct,
    CASE WHEN co2 > 0 THEN gas_co2    / co2 END                        AS gas_pct,
    CASE WHEN co2 > 0 THEN cement_co2 / co2 END                        AS cement_pct,
    -- Paris Agreement treatment.
    -- ratification_year is NULL for countries that never ratified — keep NULL explicit.
    -- paris_treated and years_since_ratification are filled in upstream by silver_clean_spark.py;
    -- we do NOT COALESCE here, so a silent join failure surfaces as a not_null test breach.
    CAST(ratification_year AS INTEGER)                                 AS ratification_year,
    CAST(paris_treated AS BOOLEAN)                                     AS paris_treated,
    CAST(years_since_ratification AS INTEGER)                          AS years_since_ratification,
    CURRENT_TIMESTAMP                                                  AS _built_at
FROM base
WHERE iso_code IS NOT NULL
  AND year IS NOT NULL
