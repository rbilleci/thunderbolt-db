\echo === psql extended sql prepared bind crosstabview ===
CREATE TABLE ext_sql_exec_crosstab_people (id INT, name TEXT, bucket TEXT);
INSERT INTO ext_sql_exec_crosstab_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE ext_sql_exec_crosstab_lookup(int4) AS SELECT bucket, name, id FROM ext_sql_exec_crosstab_people WHERE id >= $1 ORDER BY id;
EXECUTE ext_sql_exec_crosstab_lookup($1) \bind 1 \crosstabview
DEALLOCATE PREPARE ext_sql_exec_crosstab_lookup;
SELECT name FROM ext_sql_exec_crosstab_people WHERE id = 2;
