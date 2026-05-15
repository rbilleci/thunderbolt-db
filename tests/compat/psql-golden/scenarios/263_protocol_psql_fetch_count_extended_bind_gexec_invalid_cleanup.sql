\echo === psql fetch count extended bind gexec invalid cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_gexec_invalid_people (id INT, name TEXT);
CREATE TABLE fetch_count_ext_bind_gexec_invalid_commands (id INT, command TEXT);
INSERT INTO fetch_count_ext_bind_gexec_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_ext_bind_gexec_invalid_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_ext_bind_gexec_invalid_people WHERE id = 2;');
SELECT command FROM fetch_count_ext_bind_gexec_invalid_commands WHERE id = $1 \bind not_an_int \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT name FROM fetch_count_ext_bind_gexec_invalid_people WHERE id = $1 \bind 3 \g
