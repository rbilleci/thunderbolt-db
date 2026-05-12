\echo === pg_catalog public namespace metadata ===
SELECT oid, nspname FROM pg_catalog.pg_namespace WHERE nspname = 'public' ORDER BY oid;
