\echo === psql SQL prepare casted argument literals ===
CREATE TABLE sql_prepare_casted_people (id INT, name TEXT);
INSERT INTO sql_prepare_casted_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepare_casted_lookup(int4, text) AS SELECT id, name FROM sql_prepare_casted_people WHERE id = $1 AND name = $2;
EXECUTE sql_prepare_casted_lookup(2::int4, 'Linus'::text);
EXECUTE sql_prepare_casted_lookup((3::pg_catalog.int4), ('Grace')::pg_catalog.text);
EXECUTE sql_prepare_casted_lookup(1::integer, 'Missing'::pg_catalog.text);
DEALLOCATE PREPARE sql_prepare_casted_lookup;
EXECUTE sql_prepare_casted_lookup(2::int4, 'Linus'::text);
SELECT name FROM sql_prepare_casted_people WHERE id = 1;
