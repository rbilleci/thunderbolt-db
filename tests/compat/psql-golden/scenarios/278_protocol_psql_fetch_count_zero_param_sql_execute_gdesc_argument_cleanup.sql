\echo === psql fetch count zero-param sql execute gdesc argument cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_gdesc_arity_people (id INT, name TEXT);
INSERT INTO fetch_count_zero_gdesc_arity_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_zero_gdesc_arity_lookup AS SELECT id, name FROM fetch_count_zero_gdesc_arity_people WHERE id >= 2 ORDER BY id;
EXECUTE fetch_count_zero_gdesc_arity_lookup(1) \gdesc
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_gdesc_arity_lookup('extra', 'args') \gdesc
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_gdesc_arity_lookup \gdesc
\set FETCH_COUNT 0
EXECUTE fetch_count_zero_gdesc_arity_lookup();
DEALLOCATE PREPARE fetch_count_zero_gdesc_arity_lookup;
SELECT name FROM fetch_count_zero_gdesc_arity_people WHERE id = 1;
