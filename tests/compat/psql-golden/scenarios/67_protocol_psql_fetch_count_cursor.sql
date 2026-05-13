\echo === psql fetch count cursor flow ===
\set FETCH_COUNT 2
CREATE TABLE fetch_people (id INT, name TEXT);
INSERT INTO fetch_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM fetch_people ORDER BY id;
