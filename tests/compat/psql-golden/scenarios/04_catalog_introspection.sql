\echo === catalog table introspection ===
CREATE TABLE catalog_people (id INT, name TEXT);
CREATE TABLE catalog_teams (id INT);
SELECT relname FROM pg_catalog.pg_class WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' ORDER BY relname;
\echo === catalog column introspection ===
SELECT attname, atttypid FROM pg_catalog.pg_attribute WHERE attrelid = 'catalog_people'::regclass AND attnum > 0 ORDER BY attnum;
