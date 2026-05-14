\echo === psql SQL prepare quoted as keyword ===
CREATE TABLE sql_prepare_as_people (id INT, name TEXT);
INSERT INTO sql_prepare_as_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE "lookup as stmt"(int4) AS SELECT name FROM sql_prepare_as_people WHERE id = $1;
EXECUTE "lookup as stmt"(2);
DEALLOCATE PREPARE "lookup as stmt";
EXECUTE "lookup as stmt"(2);
SELECT name FROM sql_prepare_as_people WHERE id = 3;
