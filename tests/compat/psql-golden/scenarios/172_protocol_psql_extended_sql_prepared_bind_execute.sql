\echo === psql extended sql prepared bind execute ===
CREATE TABLE ext_sql_exec_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_lookup($1, $2) \bind 2 Linus \gdesc
EXECUTE ext_sql_exec_lookup($1, $2) \bind 2 Linus \g
EXECUTE ext_sql_exec_lookup($1, $2) \bind nope Linus \g
EXECUTE ext_sql_exec_lookup($1, $2) \bind 3 Grace \g
DEALLOCATE PREPARE ext_sql_exec_lookup;
SELECT name FROM ext_sql_exec_people WHERE id = 1;
