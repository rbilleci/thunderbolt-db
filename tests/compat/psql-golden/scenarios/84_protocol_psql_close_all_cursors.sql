\echo === psql close all cursor lifecycle ===
CREATE TABLE close_all_people (id INT, name TEXT);
INSERT INTO close_all_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE close_all_cursor CURSOR FOR SELECT id, name FROM close_all_people ORDER BY id;
FETCH FORWARD 1 FROM close_all_cursor;
CLOSE ALL;
FETCH FORWARD 1 FROM close_all_cursor;
SELECT name FROM close_all_people WHERE id = 3;
