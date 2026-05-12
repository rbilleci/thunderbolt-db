\echo === pg_catalog pg_class relation subset metadata ===
CREATE TABLE pg_class_subset_people (id INT, name TEXT);
CREATE TABLE pg_class_subset_teams (id INT);
CREATE TABLE pg_class_subset_ignored (id INT);
SELECT c.oid, n.nspname, c.relname, c.relkind, c.relpersistence FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname IN ('pg_class_subset_people', 'pg_class_subset_teams') AND c.relkind = 'r' ORDER BY c.relname;
