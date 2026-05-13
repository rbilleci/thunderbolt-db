\echo === psql extended casted bind select ===
CREATE TABLE ext_cast_people (id INT, name TEXT);
INSERT INTO ext_cast_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_cast_people WHERE id = $1::int4 OR name = $2::text ORDER BY id \bind 2 Ada \g
SELECT id, name FROM ext_cast_people WHERE id = $1::pg_catalog.int4 AND name = $2::pg_catalog.text \bind not-an-int Linus \g
SELECT id, name FROM ext_cast_people WHERE id = $1::pg_catalog.int4 AND name = $2::pg_catalog.text \bind 2 Linus \g
