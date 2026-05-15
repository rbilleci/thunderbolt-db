\echo === psql extended sql prepared describe literal error ===
CREATE TABLE ext_sql_exec_describe_error_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_describe_error_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE ext_sql_exec_describe_error_lookup(int4) AS SELECT name FROM ext_sql_exec_describe_error_people WHERE id = $1;
EXECUTE ext_sql_exec_describe_error_lookup('not-an-int') \gdesc
EXECUTE ext_sql_exec_describe_error_lookup(2) \gdesc
EXECUTE ext_sql_exec_describe_error_lookup(2);
DEALLOCATE PREPARE ext_sql_exec_describe_error_lookup;
SELECT name FROM ext_sql_exec_describe_error_people WHERE id = 1;
