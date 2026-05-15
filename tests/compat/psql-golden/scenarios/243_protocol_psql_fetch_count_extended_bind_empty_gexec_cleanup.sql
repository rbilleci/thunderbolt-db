\echo === psql fetch count extended bind empty gexec cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_ext_bind_empty_gexec_commands (id INT, command TEXT);
INSERT INTO fetch_count_ext_bind_empty_gexec_commands (id, command) VALUES (1, 'SELECT command FROM fetch_count_ext_bind_empty_gexec_commands WHERE id = 1;'), (2, 'SELECT command FROM fetch_count_ext_bind_empty_gexec_commands WHERE id = 2;');
SELECT command FROM fetch_count_ext_bind_empty_gexec_commands WHERE id > $1 ORDER BY id \bind 99 \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
SELECT command FROM fetch_count_ext_bind_empty_gexec_commands WHERE id = $1 \bind 1 \g
