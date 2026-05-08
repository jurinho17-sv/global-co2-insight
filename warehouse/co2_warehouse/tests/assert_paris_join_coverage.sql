-- Detects silent join failure between Silver and the Paris ratifications bronze.
-- If the join ever drops to ~0% treated rows, this test surfaces it.
--
-- The 30% threshold was specified by the project owner; with the current
-- 40-country curated ratification list and a 1960-2024 panel the realistic
-- ceiling is ~12% (post-2016 country-years for ratifying countries divided by
-- total country-years), so this test will currently flag a warning until either
-- (a) the ratification list is expanded toward the full UNFCCC roster, or
-- (b) the threshold in this file is lowered to a realistic floor (~1-3%).
{{ config(severity='warn') }}

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
   OR (treated_rows::DOUBLE / total_rows) < 0.30
