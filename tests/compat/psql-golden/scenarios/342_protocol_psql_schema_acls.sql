\echo === bounded schema ACL metadata ===
GRANT USAGE, CREATE ON SCHEMA public TO PUBLIC;
\dn+ public
SELECT n.nspname, n.nspacl FROM pg_catalog.pg_namespace n WHERE n.nspname = 'public' ORDER BY n.nspname;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT ALL PRIVILEGES ON SCHEMA public TO postgres;
\dn+ public
SELECT n.nspname, n.nspacl FROM pg_catalog.pg_namespace n WHERE n.nspname = 'public' ORDER BY n.nspname;
REVOKE USAGE ON SCHEMA public FROM PUBLIC;
REVOKE ALL PRIVILEGES ON SCHEMA public FROM postgres;
\dn+ public
GRANT USAGE ON SCHEMA private TO PUBLIC;
GRANT USAGE ON SCHEMA public TO missing_role;
GRANT USAGE ON SCHEMA public TO PUBLIC WITH GRANT OPTION;
GRANT USAGE ON SCHEMA public TO PUBLIC;
DROP SCHEMA public;
CREATE SCHEMA public;
\dn+ public
