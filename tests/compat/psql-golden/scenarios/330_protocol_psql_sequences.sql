\echo === psql bounded sequence catalog objects ===
CREATE SEQUENCE public.seq_people;
CREATE SEQUENCE seq_teams;
COMMENT ON SEQUENCE public.seq_people IS 'people ids';
ALTER SEQUENCE public.seq_people RENAME TO seq_person_ids;
\ds
\ds+
SELECT c.oid, n.nspname, c.relname, c.relkind, c.relpersistence
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 's'
ORDER BY c.relname;
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','v','s')
ORDER BY c.relname, d.objsubid;
\dd seq_*
SELECT currval('public.seq_person_ids'::regclass);
SELECT nextval('public.seq_person_ids'::regclass);
SELECT currval('seq_person_ids'::regclass);
SELECT setval('public.seq_person_ids', 10, false);
SELECT nextval('seq_person_ids'::regclass);
SELECT pg_catalog.setval('public.seq_person_ids', 20);
SELECT nextval('seq_person_ids'::regclass);
SELECT last_value, is_called FROM public.seq_person_ids;
DROP SEQUENCE seq_missing;
\ds
ALTER SEQUENCE seq_missing RENAME TO seq_archived;
ALTER SEQUENCE seq_person_ids RENAME TO seq_teams;
CREATE TABLE seq_table_conflict (id INT);
ALTER SEQUENCE seq_teams RENAME TO seq_table_conflict;
ALTER SEQUENCE seq_table_conflict RENAME TO seq_after_table;
DROP SEQUENCE public.seq_person_ids, public.seq_teams;
\ds
CREATE TABLE seq_table_target (id INT);
DROP SEQUENCE seq_table_target;
CREATE SEQUENCE seq_table_target;
CREATE SEQUENCE seq_optioned START WITH 10;
DROP SEQUENCE IF EXISTS seq_missing, seq_table_target;
\ds
