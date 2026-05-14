\echo === psql move cursor lifecycle ===
CREATE TABLE move_people (id INT, name TEXT);
INSERT INTO move_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE move_people_cursor CURSOR FOR SELECT id, name FROM move_people ORDER BY id;
FETCH NEXT FROM move_people_cursor;
MOVE FORWARD 2 FROM move_people_cursor;
FETCH NEXT FROM move_people_cursor;
MOVE ALL FROM move_people_cursor;
FETCH NEXT FROM move_people_cursor;
CLOSE move_people_cursor;
DECLARE move_backwards_people CURSOR FOR SELECT id, name FROM move_people ORDER BY id;
MOVE BACKWARD 1 FROM move_backwards_people;
SELECT name FROM move_people WHERE id = 2;
