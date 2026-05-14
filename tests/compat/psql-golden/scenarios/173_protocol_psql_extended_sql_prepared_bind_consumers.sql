\echo === psql extended sql prepared bind consumers ===
CREATE TABLE ext_sql_exec_vars (id INT, name TEXT);
INSERT INTO ext_sql_exec_vars (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_vars_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_vars WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_vars_lookup($1, $2) \bind 1 Ada \gset sql_exec_
\echo :sql_exec_id :sql_exec_name
EXECUTE ext_sql_exec_vars_lookup($1, $2) \bind 3 Grace \gx
DEALLOCATE PREPARE ext_sql_exec_vars_lookup;
SELECT name FROM ext_sql_exec_vars WHERE id = 2;
