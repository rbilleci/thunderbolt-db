\echo === psql cursor numeric in-marker variants ===
CREATE TABLE cursor_numeric_in_people (id INT, name TEXT);
INSERT INTO cursor_numeric_in_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
DECLARE fetch_numeric_in_cursor CURSOR FOR SELECT id, name FROM cursor_numeric_in_people ORDER BY id;
FETCH 0 IN fetch_numeric_in_cursor;
FETCH 2 IN fetch_numeric_in_cursor;
FETCH ALL IN fetch_numeric_in_cursor;
CLOSE fetch_numeric_in_cursor;
DECLARE move_numeric_in_cursor CURSOR FOR SELECT id, name FROM cursor_numeric_in_people ORDER BY id;
MOVE 0 IN move_numeric_in_cursor;
MOVE 2 IN move_numeric_in_cursor;
FETCH ALL IN move_numeric_in_cursor;
CLOSE move_numeric_in_cursor;
SELECT name FROM cursor_numeric_in_people WHERE id = 4;
