\echo === psql fetch count extended bind gdesc cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gdesc_people (id INT, name TEXT);
INSERT INTO fetch_count_ext_bind_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM fetch_count_ext_bind_gdesc_people WHERE id > $1 ORDER BY id \bind 1 \gdesc
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gdesc_people WHERE id = $1 \bind 3 \g
