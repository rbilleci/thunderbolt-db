\echo === psql SQL prepare relational select ===
CREATE TABLE sql_prepare_people (id INT, name TEXT);
INSERT INTO sql_prepare_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
PREPARE sql_prepare_lookup(int4, text) AS SELECT id, name FROM sql_prepare_people WHERE id >= $1 AND name = $2 ORDER BY id;
EXECUTE sql_prepare_lookup(2, 'Linus');
EXECUTE sql_prepare_lookup('not-an-int', 'Linus');
EXECUTE sql_prepare_lookup(1, 'Missing');
DEALLOCATE sql_prepare_lookup;
EXECUTE sql_prepare_lookup(2, 'Linus');
SELECT name FROM sql_prepare_people WHERE id = 4;
