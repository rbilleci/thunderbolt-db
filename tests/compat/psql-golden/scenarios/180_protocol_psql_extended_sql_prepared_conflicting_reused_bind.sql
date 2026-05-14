\echo === psql extended sql prepared conflicting reused bind ===
CREATE TABLE ext_sql_exec_conflict_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_conflict_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE ext_sql_exec_conflict_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_conflict_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_conflict_lookup($1, $1) \bind 2 \g
EXECUTE ext_sql_exec_conflict_lookup($1, 'Linus') \bind 2 \g
DEALLOCATE PREPARE ext_sql_exec_conflict_lookup;
SELECT name FROM ext_sql_exec_conflict_people WHERE id = 1;
