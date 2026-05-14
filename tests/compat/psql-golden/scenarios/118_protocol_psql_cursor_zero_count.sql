\echo === psql cursor zero-count movement ===
CREATE TABLE cursor_zero_people (id INT, name TEXT);
INSERT INTO cursor_zero_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE zero_people NO SCROLL CURSOR FOR SELECT id, name FROM cursor_zero_people ORDER BY id;
FETCH FORWARD 0 FROM zero_people;
FETCH FORWARD 1 FROM zero_people;
MOVE FORWARD 0 FROM zero_people;
FETCH FORWARD 1 FROM zero_people;
MOVE FORWARD 0 FROM zero_people;
FETCH ALL FROM zero_people;
CLOSE zero_people;
SELECT name FROM cursor_zero_people WHERE id = 3;
