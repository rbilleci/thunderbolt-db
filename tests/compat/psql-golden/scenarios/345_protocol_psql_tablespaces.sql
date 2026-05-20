\pset pager off
\echo === tablespaces ===
CREATE TABLESPACE appspace LOCATION '/tmp/gpu-db-appspace';
COMMENT ON TABLESPACE appspace IS 'application storage';
\db
\db+
SELECT oid, spcname, pg_catalog.pg_tablespace_location(oid) AS location FROM pg_catalog.pg_tablespace ORDER BY spcname;
CREATE TABLESPACE appspace LOCATION '/tmp/other';
DROP TABLESPACE pg_default;
DROP TABLESPACE missing_space;
DROP TABLESPACE IF EXISTS appspace, missing_space;
\db
