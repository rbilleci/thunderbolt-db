\pset pager off
\echo === shared object ACL metadata ===
CREATE ROLE acl_reader;
CREATE DATABASE acldb;
GRANT CONNECT, TEMPORARY ON DATABASE acldb TO acl_reader;
GRANT CONNECT ON DATABASE acldb TO PUBLIC;
\l+
SELECT datname, pg_catalog.array_to_string(datacl, E'\n') AS acl FROM pg_catalog.pg_database ORDER BY datname;
CREATE TABLESPACE aclspace LOCATION '/tmp/gpu-db-aclspace';
GRANT CREATE ON TABLESPACE aclspace TO acl_reader;
GRANT ALL PRIVILEGES ON TABLESPACE aclspace TO PUBLIC;
\db+
SELECT spcname, pg_catalog.array_to_string(spcacl, E'\n') AS acl FROM pg_catalog.pg_tablespace ORDER BY spcname;
ALTER ROLE acl_reader RENAME TO acl_owner;
\l+
\db+
REVOKE TEMP ON DATABASE acldb FROM acl_owner;
REVOKE CREATE ON TABLESPACE aclspace FROM PUBLIC;
\l+
\db+
GRANT SELECT ON DATABASE acldb TO PUBLIC;
GRANT CONNECT ON DATABASE missing_db TO PUBLIC;
GRANT CONNECT ON DATABASE acldb TO missing_role;
GRANT CONNECT ON DATABASE postgres TO PUBLIC;
GRANT CREATE ON TABLESPACE missing_space TO PUBLIC;
GRANT CREATE ON TABLESPACE pg_default TO PUBLIC;
GRANT SELECT ON TABLESPACE aclspace TO PUBLIC;
GRANT CONNECT ON DATABASE acldb TO PUBLIC WITH GRANT OPTION;
DROP ROLE acl_owner;
REVOKE CONNECT ON DATABASE acldb FROM acl_owner;
REVOKE CREATE ON TABLESPACE aclspace FROM acl_owner;
DROP ROLE acl_owner;
DROP DATABASE acldb;
DROP TABLESPACE aclspace;
\l
\db
