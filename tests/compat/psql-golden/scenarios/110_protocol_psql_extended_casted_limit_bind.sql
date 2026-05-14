\echo === psql extended casted limit bind select ===
CREATE TABLE ext_cast_limit_people (id INT, name TEXT);
INSERT INTO ext_cast_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_cast_limit_people WHERE id >= $1::pg_catalog.int4 ORDER BY id LIMIT $2::pg_catalog.int4 \bind 1 not-an-int \g
SELECT id, name FROM ext_cast_limit_people WHERE id >= $1::pg_catalog.int4 ORDER BY id LIMIT $2::pg_catalog.int4 \bind 2 2 \g
