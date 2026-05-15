\echo === psql fetch count duplicate cursor recovery ===
CREATE TABLE fetch_count_duplicate_cursor_people (id INT, name TEXT);
INSERT INTO fetch_count_duplicate_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
BEGIN;
DECLARE _psql_cursor CURSOR FOR SELECT id, name FROM fetch_count_duplicate_cursor_people ORDER BY id;
\set FETCH_COUNT 1
SELECT name FROM fetch_count_duplicate_cursor_people WHERE id >= 2 ORDER BY id;
\set FETCH_COUNT 0
FETCH FORWARD 1 FROM _psql_cursor;
CLOSE _psql_cursor;
COMMIT;
SELECT name FROM fetch_count_duplicate_cursor_people WHERE id = 3;
