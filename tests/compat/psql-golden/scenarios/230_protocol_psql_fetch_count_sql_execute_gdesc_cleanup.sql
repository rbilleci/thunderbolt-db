\echo === psql fetch count sql execute gdesc cleanup ===
\set FETCH_COUNT 2
CREATE TABLE fetch_count_sql_gdesc_people (id INT, name TEXT);
INSERT INTO fetch_count_sql_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE fetch_count_sql_gdesc_lookup(int4) AS SELECT id, name FROM fetch_count_sql_gdesc_people WHERE id >= $1 ORDER BY id;
EXECUTE fetch_count_sql_gdesc_lookup(2) \gdesc
\set FETCH_COUNT 0
FETCH ALL IN _psql_cursor;
EXECUTE fetch_count_sql_gdesc_lookup(3);
DEALLOCATE PREPARE fetch_count_sql_gdesc_lookup;
SELECT name FROM fetch_count_sql_gdesc_people WHERE id = 1;
