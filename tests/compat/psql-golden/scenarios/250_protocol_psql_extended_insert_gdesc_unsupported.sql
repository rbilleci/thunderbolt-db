\echo === psql extended insert gdesc unsupported recovery ===
CREATE TABLE ext_insert_gdesc_boundary (id INT, name TEXT);
INSERT INTO ext_insert_gdesc_boundary (id, name) VALUES ($1, $2) \bind 1 Ada \gdesc
SELECT id, name FROM ext_insert_gdesc_boundary ORDER BY id;
INSERT INTO ext_insert_gdesc_boundary (id, name) VALUES (2, 'Linus');
SELECT id, name FROM ext_insert_gdesc_boundary ORDER BY id;
