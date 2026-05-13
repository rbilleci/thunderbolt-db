\echo === psql extended insert unsupported ===
CREATE TABLE ext_insert_boundary (id INT, name TEXT);
INSERT INTO ext_insert_boundary (id, name) VALUES ($1, $2) \bind 1 Ada \g
SELECT id, name FROM ext_insert_boundary ORDER BY id;
