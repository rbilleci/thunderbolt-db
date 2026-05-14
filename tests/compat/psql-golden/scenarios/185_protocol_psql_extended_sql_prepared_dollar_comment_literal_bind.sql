\echo === psql extended sql prepared dollar comment literal bind ===
CREATE TABLE ext_sql_exec_dollar_comment_literal_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_dollar_comment_literal_people (id, name) VALUES (1, 'Ada -- not comment'), (2, 'Grace /* not comment */'), (3, 'Linus');
PREPARE ext_sql_exec_dollar_comment_literal_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_dollar_comment_literal_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_dollar_comment_literal_lookup($tag$Ada -- not comment$tag$, $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_dollar_comment_literal_lookup($tag$Ada -- not comment$tag$, $1) \bind 1 \g
EXECUTE ext_sql_exec_dollar_comment_literal_lookup($tag$Ada -- not comment$tag$, $1) \bind not-an-int \g
EXECUTE ext_sql_exec_dollar_comment_literal_lookup($$Grace /* not comment */$$, $1) \bind 2 \g
DEALLOCATE PREPARE ext_sql_exec_dollar_comment_literal_lookup;
SELECT name FROM ext_sql_exec_dollar_comment_literal_people WHERE id = 3;
