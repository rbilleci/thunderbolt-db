\echo === psql extended sql prepared dollar cast literal bind ===
CREATE TABLE ext_sql_exec_dollar_cast_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_dollar_cast_people (id, name) VALUES (1, 'Ada::literal'), (2, 'Grace::literal'), (3, 'Linus');
PREPARE ext_sql_exec_dollar_cast_lookup(int4, text) AS SELECT id, name FROM ext_sql_exec_dollar_cast_people WHERE id = $1 AND name = $2;
EXECUTE ext_sql_exec_dollar_cast_lookup($1::int4, $tag$Ada::literal$tag$::text) \bind 1 \gdesc
EXECUTE ext_sql_exec_dollar_cast_lookup($1::int4, $tag$Ada::literal$tag$::text) \bind 1 \g
EXECUTE ext_sql_exec_dollar_cast_lookup(2::int4, $$Grace::literal$$::pg_catalog.text) \g
EXECUTE ext_sql_exec_dollar_cast_lookup($1::int4, $tag$Ada::literal$tag$::jsonb) \bind 1 \g
EXECUTE ext_sql_exec_dollar_cast_lookup($1, 'Linus') \bind 3 \g
DEALLOCATE PREPARE ext_sql_exec_dollar_cast_lookup;
SELECT name FROM ext_sql_exec_dollar_cast_people WHERE id = 1;
