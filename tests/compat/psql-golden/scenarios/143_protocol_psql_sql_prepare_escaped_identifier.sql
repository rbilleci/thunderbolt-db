\echo === psql SQL prepare escaped quoted identifier ===
CREATE TABLE sql_prepare_escaped_people (id INT, name TEXT);
INSERT INTO sql_prepare_escaped_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE "lookup ""quoted"""(int4) AS SELECT name FROM sql_prepare_escaped_people WHERE id = $1;
EXECUTE "lookup ""quoted"""(2);
DEALLOCATE PREPARE "lookup ""quoted""";
EXECUTE "lookup ""quoted"""(2);
SELECT name FROM sql_prepare_escaped_people WHERE id = 3;
