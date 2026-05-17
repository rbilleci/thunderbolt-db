\echo === relational create insert select ===
CREATE TABLE people (id INT, name TEXT);
INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grace');
SELECT id, name FROM people ORDER BY id;
\echo === relational filter order limit ===
SELECT name, id FROM people WHERE id = 2 ORDER BY name DESC LIMIT 1;
\echo === relational range filter ===
SELECT name FROM people WHERE id >= 2 ORDER BY name LIMIT 2;
\echo === relational conjunctive filter ===
SELECT id FROM people WHERE id >= 2 AND name = 'Grace' ORDER BY id DESC LIMIT 1;
\echo === relational disjunctive filter ===
SELECT id, name FROM people WHERE (id = 1) OR (name = 'Grace') ORDER BY id LIMIT 2;
\echo === relational nested boolean filter ===
SELECT id, name FROM people WHERE (id = 1 OR id = 3) AND (name = 'Ada' OR name = 'Grace') ORDER BY id;
\echo === relational in membership filter ===
SELECT id, name FROM people WHERE id IN (1, 3) ORDER BY id DESC;
\echo === relational between filter ===
SELECT id, name FROM people WHERE id BETWEEN 2 AND 3 ORDER BY id DESC;
\echo === relational prefix like filter ===
SELECT id, name FROM people WHERE name LIKE 'Gra%' ORDER BY id DESC;
\echo === relational distinct projection ===
SELECT DISTINCT name FROM people ORDER BY name DESC LIMIT 2 OFFSET 1;
\echo === relational count aggregation ===
SELECT COUNT(*) FROM people WHERE name = 'Grace';
SELECT name, COUNT(*) FROM people GROUP BY name ORDER BY name;
\echo === relational aggregate unsupported recovery ===
SELECT name, COUNT(*) FROM people ORDER BY name;
SELECT COUNT(*) FROM people WHERE id >= 2;
\echo === relational distinct unsupported recovery ===
SELECT DISTINCT name FROM people ORDER BY id;
SELECT DISTINCT id FROM people ORDER BY id LIMIT 1;
\echo === relational ordered limit offset ===
SELECT id, name FROM people ORDER BY id LIMIT 1 OFFSET 1;
\echo === relational negative offset recovery ===
SELECT id FROM people ORDER BY id OFFSET -1;
SELECT id FROM people ORDER BY id LIMIT 1 OFFSET 2;
