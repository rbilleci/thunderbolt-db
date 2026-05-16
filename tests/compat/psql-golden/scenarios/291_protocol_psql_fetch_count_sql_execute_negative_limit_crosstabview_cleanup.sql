\echo === psql fetch count sql execute negative limit crosstabview cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_negative_crosstab_people (bucket TEXT, name TEXT, id INT);
INSERT INTO fetch_count_negative_crosstab_people (bucket, name, id) VALUES ('core', 'Ada', 1), ('core', 'Linus', 2), ('edge', 'Grace', 3);
PREPARE fetch_count_negative_crosstab_lookup(int4) AS SELECT bucket, name, id FROM fetch_count_negative_crosstab_people ORDER BY id LIMIT $1;
EXECUTE fetch_count_negative_crosstab_lookup(-1) \crosstabview
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_negative_crosstab_lookup(2) \crosstabview
\set FETCH_COUNT 0
EXECUTE fetch_count_negative_crosstab_lookup(1);
DEALLOCATE PREPARE fetch_count_negative_crosstab_lookup;
SELECT name FROM fetch_count_negative_crosstab_people WHERE id = 3;
