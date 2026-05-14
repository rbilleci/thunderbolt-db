\echo === psql cursor escaped quoted identifier lifecycle ===
CREATE TABLE cursor_escaped_identifier_people (id INT, name TEXT);
INSERT INTO cursor_escaped_identifier_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE "Escaped "" Cursor" CURSOR FOR SELECT id, name FROM cursor_escaped_identifier_people ORDER BY id;
FETCH FORWARD 1 FROM "Escaped "" Cursor";
MOVE FORWARD 1 IN "Escaped "" Cursor";
FETCH ALL "Escaped "" Cursor";
CLOSE "Escaped "" Cursor";
FETCH NEXT FROM "Escaped "" Cursor";
SELECT name FROM cursor_escaped_identifier_people WHERE id = 4;
