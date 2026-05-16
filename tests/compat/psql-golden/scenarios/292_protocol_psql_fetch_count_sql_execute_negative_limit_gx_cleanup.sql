\echo === psql fetch count sql execute negative limit gx cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_negative_gx_people (id INT, name TEXT);
INSERT INTO fetch_count_negative_gx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_negative_gx_lookup(int4) AS SELECT id, name FROM fetch_count_negative_gx_people ORDER BY id LIMIT $1;
EXECUTE fetch_count_negative_gx_lookup(-1) \gx
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_negative_gx_lookup(1) \gx
\set FETCH_COUNT 0
EXECUTE fetch_count_negative_gx_lookup(2);
DEALLOCATE PREPARE fetch_count_negative_gx_lookup;
SELECT name FROM fetch_count_negative_gx_people WHERE id = 3;
