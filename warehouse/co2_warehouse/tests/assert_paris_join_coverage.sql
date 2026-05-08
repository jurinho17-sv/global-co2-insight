-- Detects silent join failure between Silver and the Paris ratifications bronze.
-- A complete join failure shows up as 0% treated rows; any nonzero coverage
-- means the join is wired correctly. The threshold is set to 1% (well above
-- 0% but well below the realistic ~2.5-12% ceiling), which gives clean
-- detection without being mathematically infeasible — 30% would never pass
-- since the full UNFCCC roster only yields ~12% of country-years post-2016.
{{ config(severity='error') }}

WITH stats AS (
    SELECT
        COUNT(*)                                            AS total_rows,
        COUNT_IF(paris_treated = TRUE)                      AS treated_rows
    FROM {{ ref('mart_ml_features') }}
)

SELECT
    total_rows,
    treated_rows,
    ROUND(100.0 * treated_rows / NULLIF(total_rows, 0), 2)  AS treated_pct
FROM stats
WHERE total_rows = 0
   OR (treated_rows::DOUBLE / total_rows) < 0.01
