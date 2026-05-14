\echo === psql cursor semicolon quoted identifier ===
CREATE TABLE cursor_semicolon_people (id INT, name TEXT);
INSERT INTO cursor_semicolon_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE "semi;cursor" CURSOR FOR SELECT id, name FROM cursor_semicolon_people ORDER BY id;
FETCH NEXT FROM "semi;cursor";
MOVE 1 FROM "semi;cursor";
FETCH ALL FROM "semi;cursor";
CLOSE "semi;cursor";
FETCH NEXT FROM "semi;cursor";
SELECT name FROM cursor_semicolon_people WHERE id = 2;
