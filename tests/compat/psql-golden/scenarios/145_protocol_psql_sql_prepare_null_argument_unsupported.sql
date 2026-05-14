\echo === psql SQL prepare unsupported NULL argument ===
CREATE TABLE sql_prepare_null_people (id INT, name TEXT);
INSERT INTO sql_prepare_null_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepare_null_lookup(int4, text) AS SELECT id, name FROM sql_prepare_null_people WHERE id = $1 AND name = $2 ORDER BY id;
EXECUTE sql_prepare_null_lookup(NULL, 'Linus');
EXECUTE sql_prepare_null_lookup(2, 'Linus');
DEALLOCATE sql_prepare_null_lookup;
EXECUTE sql_prepare_null_lookup(2, 'Linus');
SELECT name FROM sql_prepare_null_people WHERE id = 3;
