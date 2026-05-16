\echo === psql fetch count sql execute gexec argument count cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_gexec_arity_exec_people (id INT, name TEXT);
CREATE TABLE fetch_count_gexec_arity_exec_commands (id INT, marker TEXT, command TEXT);
INSERT INTO fetch_count_gexec_arity_exec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_gexec_arity_exec_commands (id, marker, command) VALUES (1, 'run', 'SELECT name FROM fetch_count_gexec_arity_exec_people WHERE id = 2;');
PREPARE fetch_count_gexec_arity_exec_lookup(int4, text) AS SELECT command FROM fetch_count_gexec_arity_exec_commands WHERE id >= $1 AND marker = $2 ORDER BY id;
EXECUTE fetch_count_gexec_arity_exec_lookup(1) \gexec
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_gexec_arity_exec_lookup(1, 'run', 'extra') \gexec
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_gexec_arity_exec_lookup(1, 'run') \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_gexec_arity_exec_lookup(1, 'run');
DEALLOCATE PREPARE fetch_count_gexec_arity_exec_lookup;
SELECT name FROM fetch_count_gexec_arity_exec_people WHERE id = 3;
