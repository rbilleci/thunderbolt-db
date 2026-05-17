\echo === psql extended dml supported ===
CREATE TABLE ext_insert_boundary (id INT, name TEXT);
INSERT INTO ext_insert_boundary (id, name) VALUES ($1, $2) \bind 1 Ada \g
UPDATE ext_insert_boundary SET name = $1 WHERE id = $2 \bind Grace 1 \g
DELETE FROM ext_insert_boundary WHERE name = $1 \bind Missing \g
INSERT INTO ext_insert_boundary (id, name) VALUES ($1, $2) \bind bad Linus \g
INSERT INTO ext_insert_boundary (id, name) VALUES ($1, $2) \bind 2 Linus \g
DELETE FROM ext_insert_boundary WHERE id = $1 \bind 1 \g
SELECT id, name FROM ext_insert_boundary ORDER BY id;
