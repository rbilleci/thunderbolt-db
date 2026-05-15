\echo === psql fetch count zero-param sql execute gx cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_sql_gx_people (id INT, name TEXT);
INSERT INTO fetch_count_zero_sql_gx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_zero_sql_gx_lookup AS SELECT id, name FROM fetch_count_zero_sql_gx_people WHERE id = 2;
EXECUTE fetch_count_zero_sql_gx_lookup \gx
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_sql_gx_lookup();
DEALLOCATE PREPARE fetch_count_zero_sql_gx_lookup;
SELECT name FROM fetch_count_zero_sql_gx_people WHERE id = 1;
