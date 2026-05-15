\echo === psql gdesc unsupported cursor declaration options ===
CREATE TABLE gdesc_unsupported_cursor_people (id INT, name TEXT);
INSERT INTO gdesc_unsupported_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE gdesc_binary_people BINARY CURSOR FOR SELECT id, name FROM gdesc_unsupported_cursor_people ORDER BY id \gdesc
FETCH FORWARD 1 FROM gdesc_binary_people;
DECLARE gdesc_scroll_people SCROLL CURSOR FOR SELECT id, name FROM gdesc_unsupported_cursor_people ORDER BY id \gdesc
FETCH FORWARD 1 FROM gdesc_scroll_people;
DECLARE gdesc_supported_people NO SCROLL CURSOR FOR SELECT id, name FROM gdesc_unsupported_cursor_people ORDER BY id \gdesc
DECLARE gdesc_supported_people NO SCROLL CURSOR FOR SELECT id, name FROM gdesc_unsupported_cursor_people ORDER BY id;
FETCH FORWARD 2 FROM gdesc_supported_people;
CLOSE gdesc_supported_people;
SELECT name FROM gdesc_unsupported_cursor_people WHERE id = 2;
