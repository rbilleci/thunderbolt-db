\echo === psql extended sql prepared limit bind ===
CREATE TABLE ext_sql_exec_limit_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Ada'), (3, 'Ada'), (4, 'Grace');
PREPARE ext_sql_exec_limit_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_limit_people WHERE name = $1 ORDER BY id LIMIT $2;
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Ada 2 \gdesc
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Ada 2 \g
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Ada nope \g
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Ada -1 \g
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Ada 0 \g
EXECUTE ext_sql_exec_limit_lookup($1, $2) \bind Grace 1 \g
DEALLOCATE PREPARE ext_sql_exec_limit_lookup;
SELECT name FROM ext_sql_exec_limit_people WHERE id = 1;
