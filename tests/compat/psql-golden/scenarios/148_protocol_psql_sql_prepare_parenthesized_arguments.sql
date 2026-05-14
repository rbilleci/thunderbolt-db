\echo === psql SQL prepare parenthesized argument literals ===
CREATE TABLE sql_prepare_parenthesized_people (id INT, name TEXT);
INSERT INTO sql_prepare_parenthesized_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE sql_prepare_parenthesized_lookup(int4, text) AS SELECT id, name FROM sql_prepare_parenthesized_people WHERE id = $1 AND name = $2;
EXECUTE sql_prepare_parenthesized_lookup((2), ('Linus'));
EXECUTE sql_prepare_parenthesized_lookup(((3)), (('Grace')));
EXECUTE sql_prepare_parenthesized_lookup((1), ('Missing'));
DEALLOCATE PREPARE sql_prepare_parenthesized_lookup;
EXECUTE sql_prepare_parenthesized_lookup((2), ('Linus'));
SELECT name FROM sql_prepare_parenthesized_people WHERE id = 1;
