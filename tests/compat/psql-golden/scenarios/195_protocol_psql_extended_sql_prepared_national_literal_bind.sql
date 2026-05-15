\echo === psql extended sql prepared national literal bind ===
CREATE TABLE ext_sql_exec_national_literal_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_national_literal_people (id, name) VALUES (1, 'Ada''s notes'), (2, 'Grace Hopper'), (3, 'Linus');
PREPARE ext_sql_exec_national_literal_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_national_literal_people WHERE name = $1 AND id >= $2 ORDER BY id;
EXECUTE ext_sql_exec_national_literal_lookup(N'Ada''s notes', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_national_literal_lookup(N'Ada''s notes', $1) \bind 1 \g
EXECUTE ext_sql_exec_national_literal_lookup(text n'Grace Hopper', $1) \bind 2 \g
EXECUTE ext_sql_exec_national_literal_lookup(N'Linus', $1) \bind nope \g
DEALLOCATE PREPARE ext_sql_exec_national_literal_lookup;
SELECT name FROM ext_sql_exec_national_literal_people WHERE id = 1;
