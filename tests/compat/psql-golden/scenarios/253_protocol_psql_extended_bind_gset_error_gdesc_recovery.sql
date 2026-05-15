\echo === psql extended bind gset error gdesc recovery ===
CREATE TABLE ext_gset_gdesc_people (id INT, name TEXT);
INSERT INTO ext_gset_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT id, name FROM ext_gset_gdesc_people ORDER BY id \bind \gset failed_
SELECT name FROM ext_gset_gdesc_people WHERE id = $1 \bind 2 \gdesc
SELECT name FROM ext_gset_gdesc_people WHERE id = $1 \bind 2 \g
