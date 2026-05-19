\echo === catalog psql list bootstrap extension ===
\pset pager off
\dx
SELECT x.tableoid, x.oid, x.extname, n.nspname, x.extrelocatable, x.extversion, x.extconfig, x.extcondition
  FROM pg_extension x
  JOIN pg_namespace n ON n.oid = x.extnamespace;
