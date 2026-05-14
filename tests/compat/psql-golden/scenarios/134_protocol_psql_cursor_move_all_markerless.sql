\echo === psql cursor move all markerless ===
CREATE TABLE cursor_move_all_people (id INT, name TEXT);
INSERT INTO cursor_move_all_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE move_all_cursor CURSOR FOR SELECT id, name FROM cursor_move_all_people ORDER BY id;
FETCH 1 FROM move_all_cursor;
MOVE ALL move_all_cursor;
FETCH NEXT move_all_cursor;
CLOSE move_all_cursor;
SELECT name FROM cursor_move_all_people WHERE id = 4;
