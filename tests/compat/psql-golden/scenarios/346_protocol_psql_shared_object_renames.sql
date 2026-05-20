\pset pager off
\echo === shared object renames ===
CREATE ROLE app_reader WITH LOGIN;
COMMENT ON ROLE app_reader IS 'read-only app';
CREATE TABLE shared_rename_people (id INT, name TEXT);
GRANT SELECT ON TABLE shared_rename_people TO app_reader;
GRANT USAGE ON SCHEMA public TO app_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_reader;
ALTER ROLE app_reader RENAME TO app_analyst;
\du
\du+
SELECT oid, rolname FROM pg_catalog.pg_roles ORDER BY 1;
\dp shared_rename_people
\dn+ public
\ddp
CREATE DATABASE appdb;
COMMENT ON DATABASE appdb IS 'application database';
SELECT oid, datname FROM pg_catalog.pg_database ORDER BY datname;
ALTER DATABASE appdb RENAME TO appdb_archive;
\l
\l+
SELECT oid, datname FROM pg_catalog.pg_database ORDER BY datname;
CREATE TABLESPACE appspace LOCATION '/tmp/gpu-db-appspace';
COMMENT ON TABLESPACE appspace IS 'application storage';
SELECT oid, spcname, pg_catalog.pg_tablespace_location(oid) AS location FROM pg_catalog.pg_tablespace ORDER BY spcname;
ALTER TABLESPACE appspace RENAME TO appspace_fast;
\db
\db+
SELECT oid, spcname, pg_catalog.pg_tablespace_location(oid) AS location FROM pg_catalog.pg_tablespace ORDER BY spcname;
ALTER ROLE postgres RENAME TO root;
ALTER ROLE missing_role RENAME TO renamed_role;
CREATE ROLE app_reader;
ALTER ROLE app_analyst RENAME TO app_reader;
ALTER DATABASE postgres RENAME TO rootdb;
ALTER DATABASE missing_db RENAME TO renamed_db;
CREATE DATABASE appdb;
ALTER DATABASE appdb_archive RENAME TO appdb;
ALTER TABLESPACE pg_default RENAME TO appspace_default;
ALTER TABLESPACE missing_space RENAME TO renamed_space;
CREATE TABLESPACE appspace LOCATION '/tmp/other';
ALTER TABLESPACE appspace_fast RENAME TO appspace;
REVOKE SELECT ON TABLE shared_rename_people FROM app_analyst;
REVOKE USAGE ON SCHEMA public FROM app_analyst;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM app_analyst;
COMMENT ON ROLE app_analyst IS NULL;
DROP ROLE app_analyst, app_reader;
DROP DATABASE appdb_archive, appdb;
DROP TABLESPACE appspace_fast, appspace;
\du
\l
\db
