\echo === psql fetch count gexec cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_gexec_people (id INT, name TEXT);
CREATE TABLE fetch_count_gexec_commands (id INT, command TEXT);
INSERT INTO fetch_count_gexec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_gexec_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_gexec_people WHERE id = 2;'), (2, 'SELECT name FROM fetch_count_gexec_people WHERE id = 3;');
SELECT command FROM fetch_count_gexec_commands ORDER BY id \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_gexec_people WHERE id = 1;
