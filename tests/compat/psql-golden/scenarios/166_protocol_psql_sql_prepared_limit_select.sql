\echo === psql SQL prepared limit parameter select ===
CREATE TABLE sql_prepared_limit_people (id INT, name TEXT);
INSERT INTO sql_prepared_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine');
PREPARE sql_prepared_limit_lookup AS SELECT id, name FROM sql_prepared_limit_people WHERE id >= $1 ORDER BY id LIMIT $2;
EXECUTE sql_prepared_limit_lookup(1, 2);
EXECUTE sql_prepared_limit_lookup(2, 'not-a-limit');
EXECUTE sql_prepared_limit_lookup(2, 1::pg_catalog.int4);
DEALLOCATE PREPARE sql_prepared_limit_lookup;
EXECUTE sql_prepared_limit_lookup(1, 2);
SELECT name FROM sql_prepared_limit_people WHERE id = 4;
