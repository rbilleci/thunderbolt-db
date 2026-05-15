\echo === psql fetch count extended bind gset empty cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gset_empty_people (id INT, name TEXT);
INSERT INTO fetch_count_ext_bind_gset_empty_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM fetch_count_ext_bind_gset_empty_people WHERE id = $1 \bind 9 \gset fetched_
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gset_empty_people WHERE id = $1 \bind 2 \gdesc
SELECT name FROM fetch_count_ext_bind_gset_empty_people WHERE id = $1 \bind 2 \g
