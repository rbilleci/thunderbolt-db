\echo === psql extended dml gdesc metadata recovery ===
CREATE TABLE ext_insert_gdesc_boundary (id INT, name TEXT);
INSERT INTO ext_insert_gdesc_boundary (id, name) VALUES ($1, $2) \bind 1 Ada \gdesc
INSERT INTO ext_insert_gdesc_boundary (id, name) VALUES ($1, $2) \bind 1 Ada \g
UPDATE ext_insert_gdesc_boundary SET name = $1 WHERE id = $2 \bind Grace 1 \gdesc
UPDATE ext_insert_gdesc_boundary SET name = $1 WHERE id = $2 \bind Grace 1 \g
DELETE FROM ext_insert_gdesc_boundary WHERE name = $1 \bind Grace \gdesc
SELECT id, name FROM ext_insert_gdesc_boundary ORDER BY id;
DELETE FROM ext_insert_gdesc_boundary WHERE name = $1 \bind Grace \g
INSERT INTO ext_insert_gdesc_boundary (id, name) VALUES ($1, $2) \bind 2 Linus \g
SELECT id, name FROM ext_insert_gdesc_boundary ORDER BY id;
