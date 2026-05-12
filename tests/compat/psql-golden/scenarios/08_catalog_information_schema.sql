\echo === information schema table and column introspection ===
CREATE TABLE info_people (id INT, name TEXT);
CREATE TABLE info_teams (id INT);
SELECT table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
SELECT table_schema, table_name, column_name, ordinal_position, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'info_people' ORDER BY ordinal_position;
