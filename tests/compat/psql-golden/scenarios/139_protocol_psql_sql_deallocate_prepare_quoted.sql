\echo === psql SQL deallocate prepare quoted name ===
CREATE TABLE sql_deallocate_prepare_people (id INT, name TEXT);
INSERT INTO sql_deallocate_prepare_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE "Sql Prep Lookup" AS SELECT name FROM sql_deallocate_prepare_people WHERE id = $1;
EXECUTE "Sql Prep Lookup"(2);
DEALLOCATE PREPARE "Sql Prep Lookup";
EXECUTE "Sql Prep Lookup"(2);
SELECT name FROM sql_deallocate_prepare_people WHERE id = 1;
