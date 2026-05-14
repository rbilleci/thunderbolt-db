\echo === psql duplicate cursor preserves original ===
CREATE TABLE duplicate_cursor_people (id INT, name TEXT);
INSERT INTO duplicate_cursor_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE duplicate_people CURSOR FOR SELECT id, name FROM duplicate_cursor_people ORDER BY id;
DECLARE duplicate_people CURSOR FOR SELECT id, name FROM duplicate_cursor_people WHERE id = 2;
FETCH FORWARD 2 FROM duplicate_people;
CLOSE duplicate_people;
SELECT name FROM duplicate_cursor_people WHERE id = 3;
