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
CREATE TABLE domain_accounts (id account_id, label public.label);
INSERT INTO domain_accounts VALUES (7, 'Ada'), (8, 'Grace');
SELECT * FROM domain_accounts ORDER BY id;
\d domain_accounts
SELECT attname, atttypid FROM pg_catalog.pg_attribute WHERE attrelid = 'domain_accounts'::regclass AND attnum > 0 ORDER BY attnum;
SELECT column_name, data_type, udt_schema, udt_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'domain_accounts' ORDER BY ordinal_position;
DROP DOMAIN account_id;
DROP TABLE domain_accounts;
DROP DOMAIN account_id, label;
\dD
