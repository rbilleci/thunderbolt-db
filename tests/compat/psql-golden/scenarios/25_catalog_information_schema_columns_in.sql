\echo === information_schema columns table-name subset ===
CREATE TABLE is_columns_people (id INT, name TEXT);
CREATE TABLE is_columns_teams (id INT);
CREATE TABLE is_columns_audit (id INT, event TEXT);
SELECT table_schema, table_name, column_name, ordinal_position, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name IN ('is_columns_people', 'is_columns_teams') ORDER BY table_name, ordinal_position;
