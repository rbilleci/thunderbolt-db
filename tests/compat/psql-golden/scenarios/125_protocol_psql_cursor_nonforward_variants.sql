\echo === psql cursor non-forward variants ===
CREATE TABLE cursor_direction_people (id INT, name TEXT);
INSERT INTO cursor_direction_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE direction_people CURSOR FOR SELECT id, name FROM cursor_direction_people ORDER BY id;
FETCH FIRST FROM direction_people;
FETCH NEXT FROM direction_people;
MOVE LAST FROM direction_people;
MOVE RELATIVE 2 FROM direction_people;
MOVE ABSOLUTE 3 FROM direction_people;
FETCH NEXT FROM direction_people;
CLOSE direction_people;
SELECT name FROM cursor_direction_people WHERE id = 4;
