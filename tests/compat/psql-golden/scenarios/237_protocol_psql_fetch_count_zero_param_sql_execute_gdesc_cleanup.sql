\echo === psql fetch count zero-param sql execute gdesc cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_zero_sql_gdesc_people (id INT, name TEXT);
INSERT INTO fetch_count_zero_sql_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_zero_sql_gdesc_lookup AS SELECT id, name FROM fetch_count_zero_sql_gdesc_people WHERE id >= 2 ORDER BY id;
EXECUTE fetch_count_zero_sql_gdesc_lookup \gdesc
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_zero_sql_gdesc_lookup();
DEALLOCATE PREPARE fetch_count_zero_sql_gdesc_lookup;
SELECT name FROM fetch_count_zero_sql_gdesc_people WHERE id = 1;
