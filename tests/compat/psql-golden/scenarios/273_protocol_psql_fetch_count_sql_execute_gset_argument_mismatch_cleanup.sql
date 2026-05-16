\echo === psql fetch count sql execute gset argument count cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_gset_arity_exec_people (id INT, name TEXT);
INSERT INTO fetch_count_gset_arity_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_gset_arity_exec_lookup(int4, text) AS SELECT id, name FROM fetch_count_gset_arity_exec_people WHERE id >= $1 AND name = $2 ORDER BY id;
EXECUTE fetch_count_gset_arity_exec_lookup(2) \gset first_
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_gset_arity_exec_lookup(2, 'Linus', 'extra') \gset extra_
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_gset_arity_exec_lookup(2, 'Linus') \gset ok_
\set FETCH_COUNT 0
\echo captured_name=:ok_name
DEALLOCATE PREPARE fetch_count_gset_arity_exec_lookup;
SELECT name FROM fetch_count_gset_arity_exec_people WHERE id = 3;
