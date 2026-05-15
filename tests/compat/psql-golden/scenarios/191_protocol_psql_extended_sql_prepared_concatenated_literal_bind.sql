\echo === psql extended sql prepared concatenated literal bind ===
CREATE TABLE ext_sql_exec_concat_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_concat_people (id, name) VALUES (1, 'Ada Lovelace'), (2, 'Grace Hopper'), (3, 'Linus');
PREPARE ext_sql_exec_concat_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_concat_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_concat_lookup('Ada'
' Lovelace', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_concat_lookup('Ada'
' Lovelace', $1) \bind 1 \g
EXECUTE ext_sql_exec_concat_lookup(text 'Grace'
' Hopper', $1) \bind 2 \g
EXECUTE ext_sql_exec_concat_lookup('Ada'
' Lovelace', $1) \bind not-an-int \g
EXECUTE ext_sql_exec_concat_lookup('Ada' 'Lovelace', $1) \bind 1 \g
EXECUTE ext_sql_exec_concat_lookup($1, 3) \bind Linus \g
DEALLOCATE PREPARE ext_sql_exec_concat_lookup;
SELECT name FROM ext_sql_exec_concat_people WHERE id = 1;
