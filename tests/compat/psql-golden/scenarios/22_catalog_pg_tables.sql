\echo === pg_catalog pg_tables supported public tables ===
CREATE TABLE pg_tables_people (id INT, name TEXT);
CREATE TABLE pg_tables_teams (id INT);
SELECT schemaname, tablename, tableowner FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename;
