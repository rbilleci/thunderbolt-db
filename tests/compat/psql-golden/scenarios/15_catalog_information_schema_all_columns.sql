\echo === information schema all-column introspection ===
CREATE TABLE all_columns_people (id INT, name TEXT);
CREATE TABLE all_columns_teams (id INT, label TEXT);
SELECT table_schema, table_name, column_name, ordinal_position, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
