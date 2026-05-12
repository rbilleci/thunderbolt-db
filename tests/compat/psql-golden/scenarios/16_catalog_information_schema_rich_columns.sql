\echo === information schema rich column introspection ===
CREATE TABLE rich_columns_people (id INT, name TEXT);
CREATE TABLE rich_columns_teams (id INT);
SELECT table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
