\echo === psql markerless omitted-count cursor movement ===
CREATE TABLE cursor_markerless_people (id INT, name TEXT);
INSERT INTO cursor_markerless_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger'), (5, 'Barbara');
DECLARE markerless_fetch_people CURSOR FOR SELECT id, name FROM cursor_markerless_people ORDER BY id;
FETCH FORWARD markerless_fetch_people;
FETCH NEXT markerless_fetch_people;
FETCH FORWARD ALL markerless_fetch_people;
CLOSE markerless_fetch_people;
DECLARE markerless_move_people CURSOR FOR SELECT id, name FROM cursor_markerless_people ORDER BY id;
MOVE FORWARD markerless_move_people;
FETCH NEXT markerless_move_people;
MOVE NEXT markerless_move_people;
FETCH NEXT markerless_move_people;
MOVE FORWARD ALL markerless_move_people;
FETCH NEXT markerless_move_people;
SELECT name FROM cursor_markerless_people WHERE id = 5;
