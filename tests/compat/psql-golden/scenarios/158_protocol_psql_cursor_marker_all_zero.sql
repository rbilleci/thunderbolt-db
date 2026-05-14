\echo === psql cursor marker all and zero variants ===
CREATE TABLE cursor_marker_all_zero_people (id INT, name TEXT);
INSERT INTO cursor_marker_all_zero_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE fetch_all_from_cursor CURSOR FOR SELECT id, name FROM cursor_marker_all_zero_people ORDER BY id;
MOVE 0 FROM fetch_all_from_cursor;
FETCH ALL FROM fetch_all_from_cursor;
FETCH NEXT FROM fetch_all_from_cursor;
CLOSE fetch_all_from_cursor;
DECLARE fetch_all_in_cursor CURSOR FOR SELECT id, name FROM cursor_marker_all_zero_people ORDER BY id;
MOVE 0 IN fetch_all_in_cursor;
FETCH ALL IN fetch_all_in_cursor;
FETCH NEXT IN fetch_all_in_cursor;
CLOSE fetch_all_in_cursor;
SELECT name FROM cursor_marker_all_zero_people WHERE id = 4;
