\echo === psql fetch count extended bind gx unsupported cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gx_people (id INT, name TEXT);
INSERT INTO fetch_count_ext_bind_gx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM fetch_count_ext_bind_gx_people WHERE id = $1 \bind 2 \gx
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT id, name FROM fetch_count_ext_bind_gx_people WHERE id = $1 \bind 1 \gx
