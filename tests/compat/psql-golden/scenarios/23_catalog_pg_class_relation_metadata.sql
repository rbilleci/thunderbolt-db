\echo === pg_catalog pg_class supported public relation metadata ===
CREATE TABLE pg_class_people (id INT, name TEXT);
CREATE TABLE pg_class_teams (id INT);
SELECT c.oid, n.nspname, c.relname, c.relkind, c.relpersistence FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY c.relname;
