\echo === psql fetch count sql execute gset cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_sql_gset_people (id INT, name TEXT);
INSERT INTO fetch_count_sql_gset_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_sql_gset_lookup(int4) AS SELECT name FROM fetch_count_sql_gset_people WHERE id = $1;
EXECUTE fetch_count_sql_gset_lookup(2) \gset fetched_
\set FETCH_COUNT 0
\echo fetched_name=:fetched_name
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_gset_lookup(3);
DEALLOCATE PREPARE fetch_count_sql_gset_lookup;
SELECT name FROM fetch_count_sql_gset_people WHERE id = 1;
