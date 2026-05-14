\echo === psql extended bind gset ===
CREATE TABLE ext_gset_people (id INT, name TEXT);
INSERT INTO ext_gset_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_gset_people WHERE id = $1 \bind 2 \gset gset_
\echo :gset_id :gset_name
SELECT name FROM ext_gset_people WHERE id = :gset_id;
