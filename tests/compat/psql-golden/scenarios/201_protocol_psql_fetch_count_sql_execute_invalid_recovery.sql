\echo === psql fetch_count sql prepared execute invalid recovery ===
CREATE TABLE ext_sql_exec_invalid_cursor_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_invalid_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
PREPARE ext_sql_exec_invalid_cursor_lookup(int4) AS SELECT id, name FROM ext_sql_exec_invalid_cursor_people WHERE id >= $1 ORDER BY id;
\set FETCH_COUNT 1
EXECUTE ext_sql_exec_invalid_cursor_lookup('bad');
EXECUTE ext_sql_exec_invalid_cursor_lookup(2);
\unset FETCH_COUNT
DEALLOCATE PREPARE ext_sql_exec_invalid_cursor_lookup;
SELECT name FROM ext_sql_exec_invalid_cursor_people WHERE id = 1;
