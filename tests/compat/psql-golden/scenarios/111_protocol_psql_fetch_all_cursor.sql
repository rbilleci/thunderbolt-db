\echo === psql fetch all cursor lifecycle ===
CREATE TABLE fetch_all_people (id INT, name TEXT);
INSERT INTO fetch_all_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE all_people CURSOR FOR SELECT id, name FROM fetch_all_people ORDER BY id;
FETCH FORWARD 1 FROM all_people;
FETCH FORWARD ALL FROM all_people;
FETCH ALL FROM all_people;
CLOSE all_people;
SELECT name FROM fetch_all_people WHERE id = 3;
