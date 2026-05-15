\echo === psql extended sql prepared parenthesized dollar literal bind ===
CREATE TABLE ext_sql_exec_parenthesized_dollar_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_parenthesized_dollar_people (id, name) VALUES (1, 'Ada (literal'), (2, 'Grace ) literal'), (3, 'Linus');
PREPARE ext_sql_exec_parenthesized_dollar_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_parenthesized_dollar_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_parenthesized_dollar_lookup(($tag$Ada (literal$tag$), $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_parenthesized_dollar_lookup(($tag$Ada (literal$tag$), $1) \bind 1 \g
EXECUTE ext_sql_exec_parenthesized_dollar_lookup(CAST(($$Grace ) literal$$) AS text), $1) \bind 2 \g
EXECUTE ext_sql_exec_parenthesized_dollar_lookup(($tag$Ada (literal$tag$), $1) \bind not-an-int \g
EXECUTE ext_sql_exec_parenthesized_dollar_lookup($1, 3) \bind Linus \g
DEALLOCATE PREPARE ext_sql_exec_parenthesized_dollar_lookup;
SELECT name FROM ext_sql_exec_parenthesized_dollar_people WHERE id = 1;
