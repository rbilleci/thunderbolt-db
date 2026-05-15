\echo === psql fetch count sql execute empty cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_empty_exec_people (id INT, name TEXT);
INSERT INTO fetch_count_empty_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_empty_exec_lookup(text) AS SELECT id, name FROM fetch_count_empty_exec_people WHERE name = $1 ORDER BY id;
EXECUTE fetch_count_empty_exec_lookup('Nobody');
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_empty_exec_lookup('Ada');
DEALLOCATE PREPARE fetch_count_empty_exec_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_empty_exec_people WHERE id = 3;
