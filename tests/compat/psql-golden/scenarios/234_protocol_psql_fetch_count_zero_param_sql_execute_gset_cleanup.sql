\echo === psql fetch count zero-param sql execute gset cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_sql_gset_people (id INT, name TEXT);
INSERT INTO fetch_count_zero_sql_gset_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_zero_sql_gset_lookup AS SELECT name FROM fetch_count_zero_sql_gset_people WHERE id = 2;
EXECUTE fetch_count_zero_sql_gset_lookup \gset fetched_
\set FETCH_COUNT 0
\echo fetched_name=:fetched_name
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_sql_gset_lookup();
DEALLOCATE PREPARE fetch_count_zero_sql_gset_lookup;
SELECT name FROM fetch_count_zero_sql_gset_people WHERE id = 1;
