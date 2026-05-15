\echo === psql fetch count crosstabview cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_crosstab_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_crosstab_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
SELECT bucket, name, id FROM fetch_count_crosstab_people ORDER BY id \crosstabview
FETCH ALL IN _psql_cursor;
SELECT id, name FROM fetch_count_crosstab_people ORDER BY id;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_crosstab_people WHERE id = 2;
