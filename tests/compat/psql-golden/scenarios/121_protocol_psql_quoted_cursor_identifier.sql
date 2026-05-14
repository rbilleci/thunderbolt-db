\echo === psql quoted cursor identifier lifecycle ===
CREATE TABLE quoted_cursor_people (id INT, name TEXT);
INSERT INTO quoted_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE "Mixed Cursor" CURSOR FOR SELECT id, name FROM quoted_cursor_people ORDER BY id;
FETCH FORWARD 1 FROM "Mixed Cursor";
FETCH NEXT FROM "mixed cursor";
MOVE FORWARD 1 "Mixed Cursor";
FETCH ALL FROM "Mixed Cursor";
CLOSE "Mixed Cursor";
SELECT name FROM quoted_cursor_people WHERE id = 4;
