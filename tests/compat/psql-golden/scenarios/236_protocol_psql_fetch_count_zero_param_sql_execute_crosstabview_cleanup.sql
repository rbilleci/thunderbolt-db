\echo === psql fetch count zero-param sql execute crosstabview cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_sql_crosstab_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_zero_sql_crosstab_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE fetch_count_zero_sql_crosstab_lookup AS SELECT bucket, name, id FROM fetch_count_zero_sql_crosstab_people ORDER BY id;
EXECUTE fetch_count_zero_sql_crosstab_lookup \crosstabview
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_sql_crosstab_lookup();
DEALLOCATE PREPARE fetch_count_zero_sql_crosstab_lookup;
SELECT name FROM fetch_count_zero_sql_crosstab_people WHERE id = 1;
