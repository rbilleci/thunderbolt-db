\echo === psql fetch count zero-param sql execute crosstabview argument cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_crosstab_arity_people (id INT, name TEXT, bucket TEXT);
INSERT INTO fetch_count_zero_crosstab_arity_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
PREPARE fetch_count_zero_crosstab_arity_lookup AS SELECT bucket, name, id FROM fetch_count_zero_crosstab_arity_people ORDER BY id;
EXECUTE fetch_count_zero_crosstab_arity_lookup(1) \crosstabview
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_crosstab_arity_lookup('extra', 'args') \crosstabview
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_crosstab_arity_lookup \crosstabview
FETCH ALL IN _psql_cursor;
\set FETCH_COUNT 0
EXECUTE fetch_count_zero_crosstab_arity_lookup();
DEALLOCATE PREPARE fetch_count_zero_crosstab_arity_lookup;
SELECT name FROM fetch_count_zero_crosstab_arity_people WHERE id = 1;
