\echo === information_schema tables table-name subset ===
CREATE TABLE is_tables_people (id INT, name TEXT);
CREATE TABLE is_tables_teams (id INT);
CREATE TABLE is_tables_audit (id INT, event TEXT);
SELECT table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('is_tables_people', 'is_tables_teams') ORDER BY table_name;
