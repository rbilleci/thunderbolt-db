\echo === psql extended sql prepared mixed literal bind ===
CREATE TABLE ext_sql_exec_mixed_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_mixed_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_mixed_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_mixed_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_mixed_lookup($1, 'Linus') \bind 2 \gdesc
EXECUTE ext_sql_exec_mixed_lookup($1, 'Linus') \bind 2 \g
EXECUTE ext_sql_exec_mixed_lookup($1, 'Linus') \bind nope \g
EXECUTE ext_sql_exec_mixed_lookup(3, $1) \bind Grace \gdesc
EXECUTE ext_sql_exec_mixed_lookup(3, $1) \bind Grace \g
DEALLOCATE PREPARE ext_sql_exec_mixed_lookup;
SELECT name FROM ext_sql_exec_mixed_people WHERE id = 1;
