\echo === pg_catalog pg_indexes empty supported subset ===
CREATE TABLE pg_indexes_people (id INT, name TEXT);
SELECT schemaname, tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;
