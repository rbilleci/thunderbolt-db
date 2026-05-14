\echo === psql extended sql prepared compatible reused bind ===
CREATE TABLE ext_sql_exec_compatible_reuse_people (id INT, other_id INT, name TEXT);
INSERT INTO ext_sql_exec_compatible_reuse_people (id, other_id, name) VALUES (1, 1, 'Ada'), (2, 1, 'Linus'), (3, 3, 'Grace');
PREPARE ext_sql_exec_compatible_reuse_lookup(int4, int4) AS SELECT name FROM ext_sql_exec_compatible_reuse_people WHERE id = $1 AND other_id = $2;
EXECUTE ext_sql_exec_compatible_reuse_lookup($1, $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_compatible_reuse_lookup($1, $1) \bind not-an-int \g
EXECUTE ext_sql_exec_compatible_reuse_lookup($1, $1) \bind 1 \g
EXECUTE ext_sql_exec_compatible_reuse_lookup($1, $1) \bind 2 \g
DEALLOCATE PREPARE ext_sql_exec_compatible_reuse_lookup;
SELECT name FROM ext_sql_exec_compatible_reuse_people WHERE id = 3;
