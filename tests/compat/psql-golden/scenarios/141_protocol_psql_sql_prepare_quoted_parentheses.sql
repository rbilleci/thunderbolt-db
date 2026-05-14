\echo === psql SQL prepare quoted parentheses ===
CREATE TABLE sql_prepare_parentheses_people (id INT, name TEXT);
INSERT INTO sql_prepare_parentheses_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE "lookup(one)"(int4) AS SELECT name FROM sql_prepare_parentheses_people WHERE id = $1;
EXECUTE "lookup(one)"(2);
DEALLOCATE PREPARE "lookup(one)";
EXECUTE "lookup(one)"(2);
SELECT name FROM sql_prepare_parentheses_people WHERE id = 3;
