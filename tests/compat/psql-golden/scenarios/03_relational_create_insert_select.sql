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
\echo === relational sum aggregation ===
SELECT SUM(id) FROM people WHERE name = 'Grace';
SELECT name, SUM(id) FROM people GROUP BY name ORDER BY sum DESC LIMIT 2;
\echo === relational avg aggregation ===
SELECT AVG(id) FROM people WHERE name = 'Grace';
SELECT name, AVG(id) FROM people GROUP BY name ORDER BY avg DESC LIMIT 2;
\echo === relational min max aggregation ===
SELECT MIN(id) FROM people WHERE id >= 2;
SELECT MAX(name) FROM people WHERE id >= 2;
SELECT name, MIN(id) FROM people GROUP BY name ORDER BY min DESC LIMIT 2;
SELECT name, MAX(id) FROM people GROUP BY name ORDER BY max DESC LIMIT 2;
\echo === relational aggregate unsupported recovery ===
SELECT name, COUNT(*) FROM people ORDER BY name;
SELECT SUM(name) FROM people;
SELECT AVG(name) FROM people;
SELECT name, MAX(id) FROM people ORDER BY name;
SELECT COUNT(*) FROM people WHERE id >= 2;
\echo === relational distinct unsupported recovery ===
SELECT DISTINCT name FROM people ORDER BY id;
SELECT DISTINCT id FROM people ORDER BY id LIMIT 1;
\echo === relational ordered limit offset ===
SELECT id, name FROM people ORDER BY id LIMIT 1 OFFSET 1;
\echo === relational negative offset recovery ===
SELECT id FROM people ORDER BY id OFFSET -1;
SELECT id FROM people ORDER BY id LIMIT 1 OFFSET 2;
\echo === relational update workflow ===
UPDATE people SET name = 'Updated' WHERE id = 2 OR name LIKE 'Ada%';
SELECT id, name FROM people ORDER BY id;
SELECT COUNT(*) FROM people WHERE name = 'Updated';
\echo === relational update unsupported recovery ===
UPDATE people SET name = 'NoWhere';
UPDATE people SET id = 'bad' WHERE name = 'Grace';
SELECT id, name FROM people WHERE name = 'Grace' ORDER BY id;
\echo === relational delete workflow ===
DELETE FROM people WHERE id = 2 OR name LIKE 'Updated%';
SELECT id, name FROM people ORDER BY id;
SELECT COUNT(*) FROM people;
