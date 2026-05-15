\echo === psql fetch count gdesc cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_gdesc_people (id INT, name TEXT);
INSERT INTO fetch_count_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM fetch_count_gdesc_people ORDER BY id \gdesc
FETCH ALL IN _psql_cursor;
SELECT id, name FROM fetch_count_gdesc_people ORDER BY id;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_gdesc_people WHERE id = 2;
