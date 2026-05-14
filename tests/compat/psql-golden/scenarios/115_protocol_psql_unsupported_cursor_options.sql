\echo === psql unsupported cursor declaration options ===
CREATE TABLE unsupported_cursor_people (id INT, name TEXT);
INSERT INTO unsupported_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE binary_people BINARY CURSOR FOR SELECT id, name FROM unsupported_cursor_people ORDER BY id;
FETCH FORWARD 1 FROM binary_people;
DECLARE scroll_people SCROLL CURSOR FOR SELECT id, name FROM unsupported_cursor_people ORDER BY id;
FETCH FORWARD 1 FROM scroll_people;
DECLARE supported_people NO SCROLL CURSOR FOR SELECT id, name FROM unsupported_cursor_people ORDER BY id;
FETCH FORWARD 2 FROM supported_people;
CLOSE supported_people;
SELECT name FROM unsupported_cursor_people WHERE id = 2;
