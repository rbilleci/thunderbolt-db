\echo === psql fetch count extended bind gexec unsupported cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gexec_people (id INT, name TEXT);
CREATE TABLE fetch_count_ext_bind_gexec_commands (id INT, command TEXT);
INSERT INTO fetch_count_ext_bind_gexec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_ext_bind_gexec_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_ext_bind_gexec_people WHERE id = 2;'), (2, 'SELECT name FROM fetch_count_ext_bind_gexec_people WHERE id = 3;');
SELECT command FROM fetch_count_ext_bind_gexec_commands WHERE id = $1 \bind 1 \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gexec_people WHERE id = $1 \bind 3 \g
