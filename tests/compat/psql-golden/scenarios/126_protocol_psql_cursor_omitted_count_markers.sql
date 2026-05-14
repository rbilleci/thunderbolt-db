\echo === psql cursor omitted-count marker variants ===
CREATE TABLE cursor_marker_people (id INT, name TEXT);
INSERT INTO cursor_marker_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE marker_fetch_people CURSOR FOR SELECT id, name FROM cursor_marker_people ORDER BY id;
FETCH FORWARD FROM marker_fetch_people;
FETCH NEXT IN marker_fetch_people;
FETCH FROM marker_fetch_people;
CLOSE marker_fetch_people;
DECLARE marker_move_people CURSOR FOR SELECT id, name FROM cursor_marker_people ORDER BY id;
MOVE FORWARD FROM marker_move_people;
FETCH NEXT FROM marker_move_people;
MOVE NEXT IN marker_move_people;
FETCH NEXT FROM marker_move_people;
SELECT name FROM cursor_marker_people WHERE id = 4;
