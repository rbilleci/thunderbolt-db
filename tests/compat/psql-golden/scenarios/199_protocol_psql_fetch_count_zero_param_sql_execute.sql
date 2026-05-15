\echo === psql fetch_count zero-parameter sql prepared execute ===
CREATE TABLE ext_sql_exec_zero_cursor_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_zero_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
PREPARE "Exec Cursor Zero" AS SELECT id, name FROM ext_sql_exec_zero_cursor_people ORDER BY id;
\set FETCH_COUNT 1
EXECUTE "Exec Cursor Zero";
\unset FETCH_COUNT
DEALLOCATE PREPARE "Exec Cursor Zero";
SELECT name FROM ext_sql_exec_zero_cursor_people WHERE id = 3;
