\echo === psql gdesc cursor declaration no-data ===
CREATE TABLE gdesc_cursor_people (id INT, name TEXT);
INSERT INTO gdesc_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
DECLARE gdesc_cursor CURSOR FOR SELECT id, name FROM gdesc_cursor_people ORDER BY id \gdesc
DECLARE gdesc_cursor CURSOR FOR SELECT id, name FROM gdesc_cursor_people ORDER BY id;
FETCH 1 FROM gdesc_cursor;
CLOSE gdesc_cursor;
SELECT name FROM gdesc_cursor_people WHERE id = 2;
