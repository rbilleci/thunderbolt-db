\echo === catalog psql list bootstrap extension ===
\pset pager off
CREATE EXTENSION IF NOT EXISTS plpgsql;
CREATE EXTENSION IF NOT EXISTS "plpgsql" WITH SCHEMA pg_catalog;
CREATE EXTENSION plpgsql;
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA public;
DROP EXTENSION IF EXISTS plpgsql;
\dx
COMMENT ON EXTENSION plpgsql IS 'bootstrap extension';
\dx
SELECT description, classoid, objoid, objsubid
  FROM pg_catalog.pg_description
 ORDER BY classoid, objoid, objsubid;
COMMENT ON EXTENSION plpgsql IS NULL;
\dx
COMMENT ON EXTENSION hstore IS 'missing';
SELECT x.tableoid, x.oid, x.extname, n.nspname, x.extrelocatable, x.extversion, x.extconfig, x.extcondition
  FROM pg_extension x
  JOIN pg_namespace n ON n.oid = x.extnamespace;
