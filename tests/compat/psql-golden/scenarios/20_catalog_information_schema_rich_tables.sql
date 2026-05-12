\echo === information schema rich table introspection ===
CREATE TABLE rich_tables_people (id INT, name TEXT);
CREATE TABLE rich_tables_teams (id INT);
SELECT table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
