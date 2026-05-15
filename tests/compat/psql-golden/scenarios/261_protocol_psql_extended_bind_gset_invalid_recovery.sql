\echo === psql extended bind gset invalid recovery ===
CREATE TABLE ext_bind_gset_invalid_people (id INT, name TEXT);
INSERT INTO ext_bind_gset_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_bind_gset_invalid_people WHERE id = $1 \bind not_an_int \gset invalid_
SELECT id, name FROM ext_bind_gset_invalid_people WHERE id = $1 \bind 2 \gset valid_
\echo :valid_id :valid_name
SELECT name FROM ext_bind_gset_invalid_people WHERE id = :valid_id;
