\echo === psql extended sql prepared sparse outer bind ===
CREATE TABLE ext_sql_exec_sparse_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_sparse_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_sparse_lookup(int4, text) AS SELECT name FROM ext_sql_exec_sparse_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_sparse_lookup($2, 'Linus') \bind not-an-int-unused 2 \gdesc
EXECUTE ext_sql_exec_sparse_lookup($2, 'Linus') \bind not-an-int-unused 2 \g
EXECUTE ext_sql_exec_sparse_lookup($2, 'Linus') \bind unused not-an-int \g
EXECUTE ext_sql_exec_sparse_lookup($2, 'Grace') \bind unused 3 \g
DEALLOCATE PREPARE ext_sql_exec_sparse_lookup;
SELECT name FROM ext_sql_exec_sparse_people WHERE id = 1;
