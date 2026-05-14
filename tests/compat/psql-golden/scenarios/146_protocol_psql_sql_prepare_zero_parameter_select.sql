\echo === psql SQL prepare zero-parameter select ===
CREATE TABLE sql_prepare_noarg_people (id INT, name TEXT);
INSERT INTO sql_prepare_noarg_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepare_noarg_lookup AS SELECT id, name FROM sql_prepare_noarg_people WHERE id >= 2 ORDER BY id;
EXECUTE sql_prepare_noarg_lookup;
EXECUTE sql_prepare_noarg_lookup();
DEALLOCATE PREPARE sql_prepare_noarg_lookup;
EXECUTE sql_prepare_noarg_lookup;
SELECT name FROM sql_prepare_noarg_people WHERE id = 1;
