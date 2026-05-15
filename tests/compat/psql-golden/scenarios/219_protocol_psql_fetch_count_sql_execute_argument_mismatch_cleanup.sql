\echo === psql fetch count sql execute argument count cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_arity_exec_people (id INT, name TEXT);
INSERT INTO fetch_count_arity_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_arity_exec_lookup(int4, text) AS SELECT id, name FROM fetch_count_arity_exec_people WHERE id >= $1 AND name = $2 ORDER BY id;
EXECUTE fetch_count_arity_exec_lookup(2);
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_arity_exec_lookup(2, 'Linus', 'extra');
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_arity_exec_lookup(2, 'Linus');
DEALLOCATE PREPARE fetch_count_arity_exec_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_arity_exec_people WHERE id = 3;
