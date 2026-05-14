\echo === psql fetch next cursor lifecycle ===
CREATE TABLE fetch_next_people (id INT, name TEXT);
INSERT INTO fetch_next_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE next_people CURSOR FOR SELECT id, name FROM fetch_next_people ORDER BY id;
FETCH NEXT FROM next_people;
FETCH FORWARD FROM next_people;
FETCH 1 IN next_people;
FETCH ALL IN next_people;
CLOSE next_people;
DECLARE backwards_people CURSOR FOR SELECT id, name FROM fetch_next_people ORDER BY id;
FETCH BACKWARD 1 FROM backwards_people;
SELECT name FROM fetch_next_people WHERE id = 2;
