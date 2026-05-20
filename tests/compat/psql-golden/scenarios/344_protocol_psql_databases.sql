\echo === bounded database metadata ===
CREATE DATABASE appdb;
COMMENT ON DATABASE appdb IS 'application database';
\l
\l+
SELECT oid, datname FROM pg_catalog.pg_database ORDER BY datname;
DROP DATABASE appdb;
\l
CREATE DATABASE appdb;
CREATE DATABASE appdb;
DROP DATABASE missing_db;
DROP DATABASE IF EXISTS missing_db;
DROP DATABASE postgres;
CREATE DATABASE templated TEMPLATE template1;
COMMENT ON DATABASE missing_db IS 'bad';
DROP DATABASE appdb;
