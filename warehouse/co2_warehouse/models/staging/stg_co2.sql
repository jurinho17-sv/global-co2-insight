SELECT
    CAST(year AS INTEGER)              AS year,
    iso_code,
    country                            AS country_name,
    CAST(co2 AS DOUBLE)                AS co2,
    CAST(co2_per_capita AS DOUBLE)     AS co2_per_capita,
    co2_growth_pct,
    coal_co2,
    oil_co2,
    gas_co2,
    cement_co2,
    flaring_co2,
    primary_energy_consumption,
    gdp,
    population,
    methane,
    nitrous_oxide,
    total_ghg,
    -- WDI joins
    gdp_growth_pct,
    urban_population_pct,
    manufacturing_pct_gdp,
    -- Paris ratification join
    ratification_year,
    paris_treated,
    years_since_ratification,
    CURRENT_TIMESTAMP                  AS _loaded_at
FROM {{ source('silver_conformed', 'country_year_panel') }}
WHERE year >= 1960
  AND iso_code IS NOT NULL
  AND iso_code != ''
