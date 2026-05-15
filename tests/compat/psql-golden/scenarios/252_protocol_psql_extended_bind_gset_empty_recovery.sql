\echo === psql extended bind gset empty recovery ===
CREATE TABLE ext_gset_empty_people (id INT, name TEXT);
INSERT INTO ext_gset_empty_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT id, name FROM ext_gset_empty_people WHERE id = $1 \bind 9 \gset empty_
SELECT name FROM ext_gset_empty_people WHERE id = $1 \bind 2 \g
