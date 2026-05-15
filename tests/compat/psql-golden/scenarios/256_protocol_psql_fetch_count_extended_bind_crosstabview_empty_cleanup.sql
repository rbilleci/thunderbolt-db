\echo === psql fetch count extended bind crosstabview empty cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_crosstab_empty_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_ext_bind_crosstab_empty_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west');
SELECT bucket, name, id FROM fetch_count_ext_bind_crosstab_empty_people WHERE id > $1 ORDER BY id \bind 9 \crosstabview
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT bucket, name, id FROM fetch_count_ext_bind_crosstab_empty_people WHERE id >= $1 ORDER BY id \bind 1 \crosstabview
