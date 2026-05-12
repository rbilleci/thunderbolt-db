\echo === pg_catalog pg_attribute joined supported column types ===
CREATE TABLE pg_attribute_people (id INT, name TEXT);
SELECT a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type, a.attnotnull FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid = a.attrelid JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = 'pg_attribute_people' AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum;
