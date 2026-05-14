\echo === psql directionless numeric cursor movement ===
CREATE TABLE numeric_cursor_people (id INT, name TEXT);
INSERT INTO numeric_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE numeric_people CURSOR FOR SELECT id, name FROM numeric_cursor_people ORDER BY id;
FETCH 0 FROM numeric_people;
FETCH 2 FROM numeric_people;
MOVE 1 FROM numeric_people;
FETCH ALL FROM numeric_people;
CLOSE numeric_people;
SELECT name FROM numeric_cursor_people WHERE id = 4;
