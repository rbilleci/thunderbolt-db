\echo === psql fetch count sql execute empty gexec cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_sql_empty_gexec_people (id INT, name TEXT);
CREATE TABLE fetch_count_sql_empty_gexec_commands (id INT, command TEXT);
INSERT INTO fetch_count_sql_empty_gexec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_sql_empty_gexec_commands (id, command) VALUES (1, 'SELECT name FROM fetch_count_sql_empty_gexec_people WHERE id = 2;'), (2, 'SELECT name FROM fetch_count_sql_empty_gexec_people WHERE id = 3;');
PREPARE fetch_count_sql_empty_gexec_lookup(int4) AS SELECT command FROM fetch_count_sql_empty_gexec_commands WHERE id > $1 ORDER BY id;
EXECUTE fetch_count_sql_empty_gexec_lookup(99) \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_empty_gexec_lookup(2);
DEALLOCATE PREPARE fetch_count_sql_empty_gexec_lookup;
SELECT name FROM fetch_count_sql_empty_gexec_people WHERE id = 1;
