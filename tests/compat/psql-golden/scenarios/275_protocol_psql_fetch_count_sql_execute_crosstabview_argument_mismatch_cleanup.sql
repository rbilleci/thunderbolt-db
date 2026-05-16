\echo === psql fetch count sql execute crosstabview argument count cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_crosstab_arity_exec_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_crosstab_arity_exec_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE fetch_count_crosstab_arity_exec_lookup(int4, text) AS SELECT bucket, name, id FROM fetch_count_crosstab_arity_exec_people WHERE id >= $1 AND bucket = $2 ORDER BY id;
EXECUTE fetch_count_crosstab_arity_exec_lookup(1) \crosstabview
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_crosstab_arity_exec_lookup(1, 'east', 'extra') \crosstabview
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_crosstab_arity_exec_lookup(1, 'east') \crosstabview
\set FETCH_COUNT 0
DEALLOCATE PREPARE fetch_count_crosstab_arity_exec_lookup;
SELECT name FROM fetch_count_crosstab_arity_exec_people WHERE id = 2;
