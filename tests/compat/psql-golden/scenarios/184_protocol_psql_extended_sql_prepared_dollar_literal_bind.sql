\echo === psql extended sql prepared dollar literal bind ===
CREATE TABLE ext_sql_exec_dollar_literal_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_dollar_literal_people (id, name) VALUES (1, 'Ada, Lovelace'), (2, 'Grace (Hopper)'), (3, 'Linus');
PREPARE ext_sql_exec_dollar_literal_lookup(text, int4) AS SELECT id FROM ext_sql_exec_dollar_literal_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_dollar_literal_lookup($tag$Ada, Lovelace$tag$, $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_dollar_literal_lookup($tag$Ada, Lovelace$tag$, $1) \bind 1 \g
EXECUTE ext_sql_exec_dollar_literal_lookup($tag$Ada, Lovelace$tag$, $1) \bind not-an-int \g
EXECUTE ext_sql_exec_dollar_literal_lookup($$Grace (Hopper)$$, $1) \bind 2 \g
DEALLOCATE PREPARE ext_sql_exec_dollar_literal_lookup;
SELECT name FROM ext_sql_exec_dollar_literal_people WHERE id = 3;
