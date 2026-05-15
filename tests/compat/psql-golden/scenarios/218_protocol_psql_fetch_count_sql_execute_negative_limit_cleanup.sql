\echo === psql fetch count sql execute negative limit cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_negative_exec_people (id INT, name TEXT);
INSERT INTO fetch_count_negative_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_negative_exec_lookup(int4) AS SELECT id, name FROM fetch_count_negative_exec_people ORDER BY id LIMIT $1;
EXECUTE fetch_count_negative_exec_lookup(-1);
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_negative_exec_lookup(2);
DEALLOCATE PREPARE fetch_count_negative_exec_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_negative_exec_people WHERE id = 3;
