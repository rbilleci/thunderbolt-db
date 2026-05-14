\echo === psql cursor forward all marker variants ===
CREATE TABLE cursor_all_people (id INT, name TEXT);
INSERT INTO cursor_all_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE all_from_people CURSOR FOR SELECT id, name FROM cursor_all_people ORDER BY id;
FETCH FORWARD 1 FROM all_from_people;
MOVE FORWARD ALL FROM all_from_people;
FETCH NEXT FROM all_from_people;
CLOSE all_from_people;
DECLARE all_in_people CURSOR FOR SELECT id, name FROM cursor_all_people ORDER BY id;
MOVE FORWARD 2 IN all_in_people;
FETCH FORWARD ALL IN all_in_people;
FETCH NEXT IN all_in_people;
CLOSE all_in_people;
SELECT name FROM cursor_all_people WHERE id = 3;
