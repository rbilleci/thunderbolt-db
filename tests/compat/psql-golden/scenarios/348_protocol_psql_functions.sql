CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE sql AS 'SELECT 42';
COMMENT ON FUNCTION public.answer() IS 'metadata-only function';
\df
\df+
SELECT p.oid, n.nspname, p.proname, p.prorettype, pg_catalog.pg_get_function_result(p.oid), p.prosrc
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
ORDER BY p.proname;
SELECT p.proname, d.description
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
JOIN pg_catalog.pg_description d ON d.objoid = p.oid
WHERE n.nspname = 'public'
ORDER BY p.proname;
CREATE FUNCTION public.answer() RETURNS text LANGUAGE sql AS 'SELECT ''x''';
CREATE FUNCTION public.echo(int4) RETURNS int4 LANGUAGE sql AS 'SELECT $1';
CREATE FUNCTION public.unsupported() RETURNS bigint LANGUAGE sql AS 'SELECT 1';
CREATE FUNCTION public.unsupported_lang() RETURNS int4 LANGUAGE plpgsql AS 'BEGIN END';
COMMENT ON FUNCTION missing() IS 'missing';
DROP FUNCTION missing();
DROP FUNCTION IF EXISTS missing();
DROP FUNCTION answer();
\df
