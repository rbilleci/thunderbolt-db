\echo === psql dn schema meta-command ===
\dn
\echo === information schema schemata introspection ===
SELECT schema_name, schema_owner FROM information_schema.schemata WHERE schema_name = 'public' ORDER BY schema_name;
