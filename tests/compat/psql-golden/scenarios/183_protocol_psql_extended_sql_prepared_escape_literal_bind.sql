\echo === psql extended sql prepared escape literal bind ===
CREATE TABLE ext_sql_exec_escape_literal_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_escape_literal_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE ext_sql_exec_escape_literal_lookup(text, int4) AS SELECT id FROM ext_sql_exec_escape_literal_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_escape_literal_lookup(E'Linus', $1) \bind 2 \gdesc
EXECUTE ext_sql_exec_escape_literal_lookup(E'Linus', $1) \bind 2 \g
EXECUTE ext_sql_exec_escape_literal_lookup(E'Linus', $1) \bind not-an-int \g
EXECUTE ext_sql_exec_escape_literal_lookup(e'Grace', $1) \bind 3 \g
DEALLOCATE PREPARE ext_sql_exec_escape_literal_lookup;
SELECT name FROM ext_sql_exec_escape_literal_people WHERE id = 1;
