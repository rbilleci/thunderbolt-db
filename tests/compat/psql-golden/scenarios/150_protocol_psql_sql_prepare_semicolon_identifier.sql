\pset pager off
CREATE TABLE sql_prepare_semicolon_people (id INT, name TEXT);
INSERT INTO sql_prepare_semicolon_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE "lookup;semi"(int4) AS SELECT name FROM sql_prepare_semicolon_people WHERE id = $1;
EXECUTE "lookup;semi"(2);
DEALLOCATE PREPARE "lookup;semi";
EXECUTE "lookup;semi"(2);
SELECT name FROM sql_prepare_semicolon_people WHERE id = 3;
