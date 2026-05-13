\echo === information_schema column discovery ===
CREATE TABLE is_column_discovery_people (id INT, name TEXT);
CREATE TABLE is_column_discovery_teams (team_id INT);
SELECT table_schema, table_name, column_name, ordinal_position, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name, ordinal_position;
