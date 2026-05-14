\echo === psql extended sql prepared reused bind ===
CREATE TABLE ext_sql_exec_reused_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_reused_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_reused_lookup(int4) AS SELECT id, name FROM ext_sql_exec_reused_people WHERE id = $1 OR id = $1 ORDER BY id;
EXECUTE ext_sql_exec_reused_lookup($1) \bind 2 \gdesc
EXECUTE ext_sql_exec_reused_lookup($1) \bind 2 \g
EXECUTE ext_sql_exec_reused_lookup($1) \bind nope \g
EXECUTE ext_sql_exec_reused_lookup($1) \bind 3 \g
DEALLOCATE PREPARE ext_sql_exec_reused_lookup;
SELECT name FROM ext_sql_exec_reused_people WHERE id = 1;
