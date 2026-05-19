\echo === psql bounded public schema lifecycle ===
COMMENT ON SCHEMA public IS 'application schema';
\dn+ public
CREATE TABLE schema_lifecycle_people (id int4, name text);
DROP SCHEMA IF EXISTS public;
SELECT name FROM schema_lifecycle_people WHERE id = 1;
\dn+ public
DROP TABLE schema_lifecycle_people;
DROP SCHEMA IF EXISTS public;
SELECT schema_name, schema_owner FROM information_schema.schemata WHERE schema_name = 'public' ORDER BY schema_name;
CREATE TABLE schema_lifecycle_blocked (id int4);
CREATE SCHEMA public;
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA public;
CREATE TABLE schema_lifecycle_people (id int4, name text);
INSERT INTO schema_lifecycle_people VALUES (1, 'Ada');
SELECT name FROM schema_lifecycle_people WHERE id = 1;
\dn+ public
