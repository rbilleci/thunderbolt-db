\echo === psql fetch count gset cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_gset_people (id INT, name TEXT);
INSERT INTO fetch_count_gset_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM fetch_count_gset_people WHERE id = 2 \gset fetched_
\set FETCH_COUNT 0
\echo fetched_name=:fetched_name
FETCH ALL IN _psql_cursor;
SELECT id, name FROM fetch_count_gset_people ORDER BY id;
