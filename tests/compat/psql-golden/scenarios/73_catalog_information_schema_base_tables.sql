\echo === information_schema base table discovery ===
CREATE TABLE is_base_tables_people (id INT, name TEXT);
CREATE TABLE is_base_tables_teams (team_id INT);
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name;
