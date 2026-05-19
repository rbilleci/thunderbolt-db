\echo === bounded role metadata and ACL grantees ===
CREATE ROLE app_reader WITH LOGIN;
CREATE USER app_writer;
CREATE ROLE app_batch NOLOGIN;
COMMENT ON ROLE app_reader IS 'read-only app';
\du
\du+
SELECT oid, rolname FROM pg_catalog.pg_roles ORDER BY 1;
CREATE TABLE role_acl_people (id INT, name TEXT);
GRANT SELECT ON TABLE role_acl_people TO app_reader;
\dp role_acl_people
GRANT USAGE ON SCHEMA public TO app_reader;
\dn+ public
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_writer;
\ddp
DROP ROLE app_reader;
REVOKE SELECT ON TABLE role_acl_people FROM app_reader;
REVOKE USAGE ON SCHEMA public FROM app_reader;
COMMENT ON ROLE app_reader IS NULL;
DROP ROLE app_reader, app_batch;
\du
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM app_writer;
DROP USER app_writer;
\du
CREATE ROLE app_reader PASSWORD 'secret';
CREATE ROLE postgres;
DROP ROLE missing_role;
DROP ROLE IF EXISTS missing_role;
DROP ROLE postgres;
GRANT SELECT ON TABLE role_acl_people TO missing_role;
