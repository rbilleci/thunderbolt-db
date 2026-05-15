\echo === psql cursor negative count recovery ===
CREATE TABLE cursor_negative_count_people (id INT, name TEXT);
INSERT INTO cursor_negative_count_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE cursor_negative_count_people CURSOR FOR SELECT id, name FROM cursor_negative_count_people ORDER BY id;
FETCH 2 FROM cursor_negative_count_people;
FETCH -1 FROM cursor_negative_count_people;
FETCH 1 FROM cursor_negative_count_people;
MOVE -1 FROM cursor_negative_count_people;
MOVE 1 FROM cursor_negative_count_people;
FETCH ALL FROM cursor_negative_count_people;
CLOSE cursor_negative_count_people;
SELECT name FROM cursor_negative_count_people WHERE id = 4;
