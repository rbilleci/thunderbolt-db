\echo === psql extended sql prepared unicode literal bind ===
CREATE TABLE ext_sql_exec_unicode_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_unicode_people (id, name) VALUES (1, 'Ada Lovelace'), (2, 'Grace Hopper'), (3, 'Linus');
PREPARE ext_sql_exec_unicode_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_unicode_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_unicode_lookup(U&'Ada\0020Lovelace', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_unicode_lookup(U&'Ada\0020Lovelace', $1) \bind 1 \g
EXECUTE ext_sql_exec_unicode_lookup($1, 3) \bind Linus \g
EXECUTE ext_sql_exec_unicode_lookup(text U&'Grace\+000020Hopper', $1) \bind 2 \g
EXECUTE ext_sql_exec_unicode_lookup(U&'Ada\0020Lovelace', $1) \bind not-an-int \g
EXECUTE ext_sql_exec_unicode_lookup(U&'bad\00xz', $1) \bind 1 \g
EXECUTE ext_sql_exec_unicode_lookup($1, 3) \bind Linus \g
DEALLOCATE PREPARE ext_sql_exec_unicode_lookup;
SELECT name FROM ext_sql_exec_unicode_people WHERE id = 1;
