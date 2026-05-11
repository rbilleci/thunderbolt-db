\echo === catalog table introspection ===
CREATE TABLE catalog_people (id INT, name TEXT);
CREATE TABLE catalog_teams (id INT);
SELECT relname FROM pg_catalog.pg_class WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' ORDER BY relname;
SELECT oid, relname FROM pg_catalog.pg_class WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' ORDER BY oid;
\echo === catalog column introspection ===
SELECT attname, atttypid FROM pg_catalog.pg_attribute WHERE attrelid = 'catalog_people'::regclass AND attnum > 0 ORDER BY attnum;
SELECT attnum, attname, atttypid, attlen FROM pg_catalog.pg_attribute WHERE attrelid = 'catalog_people'::regclass AND attnum > 0 ORDER BY attnum;
\echo === catalog type introspection ===
SELECT oid, typname, typlen FROM pg_catalog.pg_type WHERE oid IN (23, 25) ORDER BY oid;
SELECT typname, oid, typlen FROM pg_catalog.pg_type WHERE typname IN ('int4', 'text') ORDER BY typname;
