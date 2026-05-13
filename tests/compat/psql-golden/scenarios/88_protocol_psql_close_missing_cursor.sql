\echo === psql close missing cursor recovery ===
CREATE TABLE close_missing_people (id INT, name TEXT);
INSERT INTO close_missing_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
DECLARE close_missing_cursor CURSOR FOR SELECT id, name FROM close_missing_people ORDER BY id;
CLOSE missing_cursor;
FETCH FORWARD 1 FROM close_missing_cursor;
SELECT name FROM close_missing_people WHERE id = 2;
