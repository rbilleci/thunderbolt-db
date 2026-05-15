\echo === psql fetch count sql execute crosstabview cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_sql_crosstab_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_sql_crosstab_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE fetch_count_sql_crosstab_lookup(int4) AS SELECT bucket, name, id FROM fetch_count_sql_crosstab_people WHERE id >= $1 ORDER BY id;
EXECUTE fetch_count_sql_crosstab_lookup(1) \crosstabview
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_crosstab_lookup(2);
DEALLOCATE PREPARE fetch_count_sql_crosstab_lookup;
SELECT name FROM fetch_count_sql_crosstab_people WHERE id = 1;
