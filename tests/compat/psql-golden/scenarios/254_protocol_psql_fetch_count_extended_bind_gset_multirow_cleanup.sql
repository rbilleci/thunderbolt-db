\echo === psql fetch count extended bind gset multirow cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gset_multi_people (id INT, name TEXT);
INSERT INTO fetch_count_ext_bind_gset_multi_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM fetch_count_ext_bind_gset_multi_people WHERE id > $1 ORDER BY id \bind 1 \gset fetched_
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gset_multi_people WHERE id = $1 \bind 3 \gdesc
SELECT name FROM fetch_count_ext_bind_gset_multi_people WHERE id = $1 \bind 3 \g
