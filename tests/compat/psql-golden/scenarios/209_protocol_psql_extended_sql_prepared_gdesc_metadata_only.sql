\echo === psql extended sql prepared gdesc metadata only ===
CREATE TABLE ext_sql_exec_gdesc_metadata_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_gdesc_metadata_people (id, name) VALUES (1, 'Ada'), (2, 'Ada'), (3, 'Grace');
PREPARE ext_sql_exec_gdesc_metadata_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_gdesc_metadata_people WHERE name = $1 ORDER BY id LIMIT $2;
EXECUTE ext_sql_exec_gdesc_metadata_lookup($1, $2) \bind Ada nope \gdesc
EXECUTE ext_sql_exec_gdesc_metadata_lookup($1, $2) \bind Ada 1 \gdesc
EXECUTE ext_sql_exec_gdesc_metadata_lookup($1, $2) \bind Ada 1 \g
DEALLOCATE PREPARE ext_sql_exec_gdesc_metadata_lookup;
SELECT name FROM ext_sql_exec_gdesc_metadata_people WHERE id = 3;
