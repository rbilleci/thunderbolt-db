\echo === relational create insert select ===
CREATE TABLE people (id INT, name TEXT);
INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM people ORDER BY id;
\echo === relational filter order limit ===
SELECT name, id FROM people WHERE id = 2 ORDER BY name DESC LIMIT 1;
