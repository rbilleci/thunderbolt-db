\echo === psql fetch count zero-param sql execute negative limit gx cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_negative_gx_people (id INT, name TEXT);
INSERT INTO fetch_count_zero_negative_gx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_zero_negative_gx_lookup AS SELECT id, name FROM fetch_count_zero_negative_gx_people ORDER BY id LIMIT -1;
EXECUTE fetch_count_zero_negative_gx_lookup \gx
FETCH ALL IN _psql_cursor;
DEALLOCATE PREPARE fetch_count_zero_negative_gx_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_zero_negative_gx_people WHERE id = 2;
