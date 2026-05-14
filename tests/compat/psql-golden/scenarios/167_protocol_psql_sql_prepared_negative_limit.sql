\echo === psql SQL prepared negative limit parameter ===
CREATE TABLE sql_prepared_negative_limit_people (id INT, name TEXT);
INSERT INTO sql_prepared_negative_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepared_negative_limit_lookup AS SELECT id, name FROM sql_prepared_negative_limit_people ORDER BY id LIMIT $1;
EXECUTE sql_prepared_negative_limit_lookup(-1);
EXECUTE sql_prepared_negative_limit_lookup(0);
EXECUTE sql_prepared_negative_limit_lookup(2);
DEALLOCATE PREPARE sql_prepared_negative_limit_lookup;
SELECT name FROM sql_prepared_negative_limit_people WHERE id = 3;
