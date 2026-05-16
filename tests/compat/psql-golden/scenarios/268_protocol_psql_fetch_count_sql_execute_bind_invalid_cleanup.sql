\echo === psql fetch count sql execute bind invalid cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_sql_exec_bind_invalid_people (id INT, name TEXT);
INSERT INTO fetch_count_sql_exec_bind_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_sql_exec_bind_invalid_lookup(int4) AS SELECT id, name FROM fetch_count_sql_exec_bind_invalid_people WHERE id >= $1 ORDER BY id;
EXECUTE fetch_count_sql_exec_bind_invalid_lookup($1) \bind not_an_int \g
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_exec_bind_invalid_lookup($1) \bind 2 \g
DEALLOCATE PREPARE fetch_count_sql_exec_bind_invalid_lookup;
SELECT name FROM fetch_count_sql_exec_bind_invalid_people WHERE id = 1;
