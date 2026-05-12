\echo === information_schema extended column metadata table filter ===
CREATE TABLE is_filtered_columns_people (id INT, name TEXT);
CREATE TABLE is_filtered_columns_teams (team_id INT);
SELECT table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'is_filtered_columns_people' ORDER BY ordinal_position;
