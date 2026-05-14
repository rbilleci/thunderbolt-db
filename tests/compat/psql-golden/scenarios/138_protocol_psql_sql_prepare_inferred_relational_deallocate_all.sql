\echo === psql SQL prepare inferred types deallocate all ===
CREATE TABLE sql_prepare_inferred_people (id INT, name TEXT);
INSERT INTO sql_prepare_inferred_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepare_inferred AS SELECT id, name FROM sql_prepare_inferred_people WHERE id = $1 AND name = $2 ORDER BY id;
EXECUTE sql_prepare_inferred(2, 'Linus');
EXECUTE sql_prepare_inferred('not-an-int', 'Linus');
EXECUTE sql_prepare_inferred(3, 'Missing');
DEALLOCATE ALL;
EXECUTE sql_prepare_inferred(2, 'Linus');
SELECT name FROM sql_prepare_inferred_people WHERE id = 1;
