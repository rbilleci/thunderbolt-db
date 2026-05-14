\echo === psql SQL prepare duplicate precedence ===
CREATE TABLE sql_prepare_duplicate_people (id INT, name TEXT);
INSERT INTO sql_prepare_duplicate_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE duplicate_lookup(int4) AS SELECT name FROM sql_prepare_duplicate_people WHERE id = $1;
PREPARE duplicate_lookup(jsonb) AS INSERT INTO sql_prepare_duplicate_people (id, name) VALUES ($1, 'Grace');
EXECUTE duplicate_lookup(2);
DEALLOCATE duplicate_lookup;
EXECUTE duplicate_lookup(2);
SELECT name FROM sql_prepare_duplicate_people WHERE id = 1;
