\echo === psql markerless numeric zero cursor movement ===
CREATE TABLE cursor_markerless_numeric_zero_people (id INT, name TEXT);
INSERT INTO cursor_markerless_numeric_zero_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE markerless_zero_fetch_cursor CURSOR FOR SELECT id, name FROM cursor_markerless_numeric_zero_people ORDER BY id;
FETCH 0 markerless_zero_fetch_cursor;
FETCH 2 markerless_zero_fetch_cursor;
FETCH ALL markerless_zero_fetch_cursor;
CLOSE markerless_zero_fetch_cursor;
DECLARE markerless_zero_move_cursor CURSOR FOR SELECT id, name FROM cursor_markerless_numeric_zero_people ORDER BY id;
MOVE 0 markerless_zero_move_cursor;
MOVE 2 markerless_zero_move_cursor;
FETCH ALL markerless_zero_move_cursor;
CLOSE markerless_zero_move_cursor;
SELECT name FROM cursor_markerless_numeric_zero_people WHERE id = 4;
