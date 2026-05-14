\echo === psql cursor movement without from in separator ===
CREATE TABLE cursor_short_people (id INT, name TEXT);
INSERT INTO cursor_short_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger'), (5, 'Barbara');
DECLARE cursor_short_fetch CURSOR FOR SELECT id, name FROM cursor_short_people ORDER BY id;
FETCH cursor_short_fetch;
FETCH FORWARD 2 cursor_short_fetch;
FETCH ALL cursor_short_fetch;
CLOSE cursor_short_fetch;
DECLARE cursor_short_move CURSOR FOR SELECT id, name FROM cursor_short_people ORDER BY id;
MOVE cursor_short_move;
MOVE FORWARD 2 cursor_short_move;
FETCH NEXT cursor_short_move;
MOVE ALL cursor_short_move;
FETCH NEXT cursor_short_move;
CLOSE cursor_short_move;
DECLARE cursor_short_backwards CURSOR FOR SELECT id, name FROM cursor_short_people ORDER BY id;
FETCH BACKWARD 1 cursor_short_backwards;
SELECT name FROM cursor_short_people WHERE id = 2;
