\echo === psql SQL prepare keyword whitespace ===
CREATE TABLE sql_prepare_whitespace_people (id INT, name TEXT);
INSERT INTO sql_prepare_whitespace_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE
whitespace_lookup
(int4, text)
AS SELECT id, name FROM sql_prepare_whitespace_people WHERE id = $1 AND name = $2;
EXECUTE
whitespace_lookup
(2, 'Linus');
DEALLOCATE
PREPARE
whitespace_lookup;
EXECUTE whitespace_lookup(2, 'Linus');
SELECT name FROM sql_prepare_whitespace_people WHERE id = 1;
