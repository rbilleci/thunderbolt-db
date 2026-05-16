\echo === psql fetch count zero-param sql execute negative limit gexec cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_zero_negative_gexec_people (id INT, name TEXT);
CREATE TABLE fetch_count_zero_negative_gexec_commands (id INT, command TEXT);
INSERT INTO fetch_count_zero_negative_gexec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_zero_negative_gexec_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_zero_negative_gexec_people WHERE id = 2;');
PREPARE fetch_count_zero_negative_gexec_lookup AS SELECT command FROM fetch_count_zero_negative_gexec_commands ORDER BY id LIMIT -1;
EXECUTE fetch_count_zero_negative_gexec_lookup \gexec
FETCH ALL IN _psql_cursor;
DEALLOCATE PREPARE fetch_count_zero_negative_gexec_lookup;
\set FETCH_COUNT 0
SELECT name FROM fetch_count_zero_negative_gexec_people WHERE id = 3;
