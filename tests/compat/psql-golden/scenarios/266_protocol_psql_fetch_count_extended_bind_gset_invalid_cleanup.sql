\echo === psql fetch count extended bind gset invalid cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gset_invalid_people (id INT, name TEXT);
INSERT INTO fetch_count_ext_bind_gset_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM fetch_count_ext_bind_gset_invalid_people WHERE id >= $1 ORDER BY id \bind not_an_int \gset fetched_
\set FETCH_COUNT 0
\echo fetched_name=:fetched_name
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gset_invalid_people WHERE id = $1 \bind 3 \gset recovered_
\echo recovered_name=:recovered_name
SELECT id, name FROM fetch_count_ext_bind_gset_invalid_people WHERE id = $1 \bind 2 \g
