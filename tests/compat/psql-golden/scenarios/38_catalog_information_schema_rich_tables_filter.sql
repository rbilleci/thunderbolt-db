\echo === information_schema rich table metadata filter ===
CREATE TABLE is_table_filter_people (id INT, name TEXT);
CREATE TABLE is_table_filter_teams (team_id INT);
SELECT table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'is_table_filter_people' ORDER BY table_name;
