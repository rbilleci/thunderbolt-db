\echo === psql cursor unsupported direction recovery ===
CREATE TABLE cursor_unsupported_direction_people (id INT, name TEXT);
INSERT INTO cursor_unsupported_direction_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE unsupported_direction_people CURSOR FOR SELECT id, name FROM cursor_unsupported_direction_people ORDER BY id;
FETCH FORWARD 1 FROM unsupported_direction_people;
FETCH BACKWARD 1 FROM unsupported_direction_people;
MOVE ABSOLUTE 2 FROM unsupported_direction_people;
FETCH NEXT FROM unsupported_direction_people;
CLOSE unsupported_direction_people;
DECLARE unsupported_scroll_people SCROLL CURSOR FOR SELECT id, name FROM cursor_unsupported_direction_people ORDER BY id;
SELECT name FROM cursor_unsupported_direction_people WHERE id = 3;
