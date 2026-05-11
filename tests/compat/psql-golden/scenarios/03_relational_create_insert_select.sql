\echo === relational create insert select ===
CREATE TABLE people (id INT, name TEXT);
INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM people ORDER BY id;
\echo === relational filter order limit ===
SELECT name, id FROM people WHERE id = 2 ORDER BY name DESC LIMIT 1;
\echo === relational range filter ===
SELECT name FROM people WHERE id >= 2 ORDER BY name LIMIT 2;
\echo === relational conjunctive filter ===
SELECT id FROM people WHERE id >= 2 AND name = 'Grace' ORDER BY id DESC LIMIT 1;
\echo === relational disjunctive filter ===
SELECT id, name FROM people WHERE id = 1 OR name = 'Grace' ORDER BY id LIMIT 2;
