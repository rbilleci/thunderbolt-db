\echo === psql extended sql prepared reordered bind ===
CREATE TABLE ext_sql_exec_reordered_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_reordered_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_reordered_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_reordered_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_reordered_lookup($2, $1) \bind Linus 2 \gdesc
EXECUTE ext_sql_exec_reordered_lookup($2, $1) \bind Linus 2 \g
EXECUTE ext_sql_exec_reordered_lookup($2, $1) \bind Linus nope \g
EXECUTE ext_sql_exec_reordered_lookup($2, $1) \bind Grace 3 \g
DEALLOCATE PREPARE ext_sql_exec_reordered_lookup;
SELECT name FROM ext_sql_exec_reordered_people WHERE id = 1;
