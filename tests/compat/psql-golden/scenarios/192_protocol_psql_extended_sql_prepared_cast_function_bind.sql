\echo === psql extended sql prepared cast function bind ===
CREATE TABLE ext_sql_exec_cast_func_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_cast_func_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
PREPARE ext_sql_exec_cast_func_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_cast_func_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_cast_func_lookup(CAST($1 AS pg_catalog.int4), CAST('Ada' AS text)) \bind 1 \gdesc
EXECUTE ext_sql_exec_cast_func_lookup(CAST($1 AS pg_catalog.int4), CAST('Ada' AS text)) \bind 1 \g
EXECUTE ext_sql_exec_cast_func_lookup(CAST('2' AS int4), CAST($1 AS pg_catalog.text)) \bind Grace \g
EXECUTE ext_sql_exec_cast_func_lookup(CAST($1 AS integer), CAST('Ada' AS text)) \bind nope \g
EXECUTE ext_sql_exec_cast_func_lookup(CAST($1 AS jsonb), 'Ada') \bind 1 \g
EXECUTE ext_sql_exec_cast_func_lookup($1, 'Linus') \bind 3 \g
DEALLOCATE PREPARE ext_sql_exec_cast_func_lookup;
SELECT name FROM ext_sql_exec_cast_func_people WHERE id = 1;
