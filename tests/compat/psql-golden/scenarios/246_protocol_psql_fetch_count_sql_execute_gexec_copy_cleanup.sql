\echo === psql fetch count sql execute gexec copy cleanup ===
\set FETCH_COUNT 1
CREATE TABLE fetch_count_sql_gexec_copy_people (id INT, name TEXT);
CREATE TABLE fetch_count_sql_gexec_copy_commands (id INT, command TEXT);
INSERT INTO fetch_count_sql_gexec_copy_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO fetch_count_sql_gexec_copy_commands (id, command) VALUES (1, 'COPY fetch_count_sql_gexec_copy_people TO STDOUT;');
PREPARE fetch_count_sql_gexec_copy_lookup AS SELECT command FROM fetch_count_sql_gexec_copy_commands ORDER BY id;
EXECUTE fetch_count_sql_gexec_copy_lookup \gexec
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_gexec_copy_lookup();
DEALLOCATE PREPARE fetch_count_sql_gexec_copy_lookup;
SELECT name FROM fetch_count_sql_gexec_copy_people WHERE id = 1;
