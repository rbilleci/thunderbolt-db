\echo === psql extended bind gx invalid recovery ===
CREATE TABLE ext_bind_gx_invalid_people (id INT, name TEXT);
INSERT INTO ext_bind_gx_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_bind_gx_invalid_people WHERE id >= $1 ORDER BY id \bind not_an_int \gx
SELECT id, name FROM ext_bind_gx_invalid_people WHERE id >= $1 ORDER BY id \bind 2 \gx
