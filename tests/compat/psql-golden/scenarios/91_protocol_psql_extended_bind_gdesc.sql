\echo === psql extended bind gdesc portal describe ===
CREATE TABLE ext_bind_gdesc_people (id INT, name TEXT);
INSERT INTO ext_bind_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM ext_bind_gdesc_people WHERE id = $1 \bind 2 \gdesc
SELECT name FROM ext_bind_gdesc_people WHERE id = $1 \bind 2 \g
