\echo === psql fetch count sql execute bind crosstabview invalid cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_sql_exec_bind_crosstab_invalid_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_sql_exec_bind_crosstab_invalid_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE fetch_count_sql_exec_bind_crosstab_invalid_lookup(int4) AS SELECT bucket, name, id FROM fetch_count_sql_exec_bind_crosstab_invalid_people WHERE id >= $1 ORDER BY id;
EXECUTE fetch_count_sql_exec_bind_crosstab_invalid_lookup($1) \bind not_an_int \crosstabview
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_exec_bind_crosstab_invalid_lookup($1) \bind 2 \crosstabview
DEALLOCATE PREPARE fetch_count_sql_exec_bind_crosstab_invalid_lookup;
SELECT name FROM fetch_count_sql_exec_bind_crosstab_invalid_people WHERE id = 1;
