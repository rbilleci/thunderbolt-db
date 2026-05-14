\echo === psql extended bind gx ===
CREATE TABLE ext_gx_people (id INT, name TEXT);
INSERT INTO ext_gx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_gx_people WHERE id = $1 \bind 2 \gx
SELECT name FROM ext_gx_people WHERE id = 3;
