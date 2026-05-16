\echo === psql fetch count zero-param sql execute negative limit crosstabview cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_negative_crosstab_people (bucket TEXT, name TEXT, id INT);
INSERT INTO fetch_count_zero_negative_crosstab_people (bucket, name, id) VALUES ('core', 'Ada', 1), ('core', 'Linus', 2), ('edge', 'Grace', 3);
PREPARE fetch_count_zero_negative_crosstab_lookup AS SELECT bucket, name, id FROM fetch_count_zero_negative_crosstab_people ORDER BY id LIMIT -1;
EXECUTE fetch_count_zero_negative_crosstab_lookup \crosstabview
FETCH ALL IN _psql_cursor;
DEALLOCATE PREPARE fetch_count_zero_negative_crosstab_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_zero_negative_crosstab_people WHERE id = 2;
