\echo === psql fetch count zero-param sql execute gexec argument cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_gexec_arity_people (id INT, name TEXT);
CREATE TABLE fetch_count_zero_gexec_arity_commands (id INT, command TEXT);
INSERT INTO fetch_count_zero_gexec_arity_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_zero_gexec_arity_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_zero_gexec_arity_people WHERE id = 2;');
PREPARE fetch_count_zero_gexec_arity_lookup AS SELECT command FROM fetch_count_zero_gexec_arity_commands ORDER BY id;
EXECUTE fetch_count_zero_gexec_arity_lookup(1) \gexec
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_gexec_arity_lookup('extra', 'args') \gexec
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_gexec_arity_lookup \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_gexec_arity_lookup();
DEALLOCATE PREPARE fetch_count_zero_gexec_arity_lookup;
SELECT name FROM fetch_count_zero_gexec_arity_people WHERE id = 3;
