\echo === information_schema empty constraint metadata ===
CREATE TABLE is_constraints_people (id INT, name TEXT);
CREATE TABLE is_constraints_teams (team_id INT);
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT table_schema, table_name, column_name, constraint_name, ordinal_position FROM information_schema.key_column_usage WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
