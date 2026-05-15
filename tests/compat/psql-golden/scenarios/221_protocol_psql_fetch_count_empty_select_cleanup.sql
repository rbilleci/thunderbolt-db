\echo === psql fetch count empty select cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_empty_select_people (id INT, name TEXT);
INSERT INTO fetch_count_empty_select_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM fetch_count_empty_select_people WHERE id > 99 ORDER BY id;
FETCH ALL IN _psql_cursor;
SELECT id, name FROM fetch_count_empty_select_people WHERE id = 2;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_empty_select_people WHERE id = 3;
