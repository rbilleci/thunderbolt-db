\echo === psql cursor move all marker variants ===
CREATE TABLE cursor_move_all_marker_people (id INT, name TEXT);
INSERT INTO cursor_move_all_marker_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine'), (5, 'Edsger');
DECLARE move_all_from_cursor CURSOR FOR SELECT id, name FROM cursor_move_all_marker_people ORDER BY id;
FETCH 2 FROM move_all_from_cursor;
MOVE ALL FROM move_all_from_cursor;
FETCH NEXT FROM move_all_from_cursor;
CLOSE move_all_from_cursor;
DECLARE move_all_in_cursor CURSOR FOR SELECT id, name FROM cursor_move_all_marker_people ORDER BY id;
MOVE 1 IN move_all_in_cursor;
MOVE ALL IN move_all_in_cursor;
FETCH NEXT IN move_all_in_cursor;
CLOSE move_all_in_cursor;
SELECT name FROM cursor_move_all_marker_people WHERE id = 5;
