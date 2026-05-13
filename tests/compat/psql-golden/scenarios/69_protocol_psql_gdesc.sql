\echo === psql gdesc extended describe flow ===
CREATE TABLE gdesc_people (id INT, name TEXT);
SELECT name, id FROM gdesc_people WHERE id = 1 \gdesc
SELECT name, id FROM gdesc_people WHERE id = 1 \g
