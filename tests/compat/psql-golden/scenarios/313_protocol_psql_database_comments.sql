\echo === bounded database comments ===
COMMENT ON DATABASE postgres IS 'primary database';
\l+
SELECT pg_catalog.shobj_description(5, 'pg_database');
COMMENT ON DATABASE postgres IS NULL;
\l+
COMMENT ON DATABASE template1 IS 'bad';
SELECT 1 AS one;
