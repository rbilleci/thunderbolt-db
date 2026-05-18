\echo === pg_catalog pg_indexes supported subset ===
CREATE TABLE pg_indexes_people (id INT, name TEXT);
CREATE INDEX pg_indexes_people_name_idx ON pg_indexes_people (name);
SELECT schemaname, tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;
