\echo === protocol psql domains ===
CREATE DOMAIN public.account_id AS int4;
CREATE DOMAIN label AS text;
COMMENT ON DOMAIN public.account_id IS 'Account id domain';
\dD
\dD+
SELECT oid, typname, typbasetype, typtype FROM pg_catalog.pg_type WHERE typtype = 'd' ORDER BY typname;
DROP DOMAIN missing_domain;
DROP DOMAIN IF EXISTS missing_domain;
CREATE DOMAIN constrained_id AS int4 CHECK (VALUE > 0);
CREATE TABLE domain_column_rejected (id account_id);
DROP DOMAIN account_id, label;
\dD
