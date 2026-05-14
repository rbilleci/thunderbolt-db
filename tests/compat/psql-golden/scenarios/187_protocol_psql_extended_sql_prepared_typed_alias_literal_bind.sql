\echo === psql extended sql prepared typed alias literal bind ===
CREATE TABLE ext_sql_exec_typed_alias_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_typed_alias_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
PREPARE ext_sql_exec_typed_alias_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_typed_alias_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_typed_alias_lookup(TEXT 'Ada', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_typed_alias_lookup(TEXT 'Ada', $1) \bind 1 \g
EXECUTE ext_sql_exec_typed_alias_lookup($1, integer '2') \bind Grace \g
EXECUTE ext_sql_exec_typed_alias_lookup($1, pg_catalog.integer '3') \bind Linus \g
EXECUTE ext_sql_exec_typed_alias_lookup(pg_catalog.text $$Ada$$, $1) \bind not-an-int \g
EXECUTE ext_sql_exec_typed_alias_lookup($1, (pg_catalog.integer '3')) \bind Linus \g
DEALLOCATE PREPARE ext_sql_exec_typed_alias_lookup;
SELECT name FROM ext_sql_exec_typed_alias_people WHERE id = 1;
