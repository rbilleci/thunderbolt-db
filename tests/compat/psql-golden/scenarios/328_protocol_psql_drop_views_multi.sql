\echo === bounded multi-view drops ===
CREATE TABLE drop_views_people (id INT, name TEXT);
INSERT INTO drop_views_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE VIEW public.active_drop_views_people AS SELECT id, name FROM drop_views_people WHERE id > 1 ORDER BY id;
CREATE VIEW public.other_drop_views_people AS SELECT id, name FROM drop_views_people WHERE id = 1;
CREATE VIEW public.third_drop_views_people AS SELECT id, name FROM drop_views_people WHERE name LIKE 'G%' ORDER BY id;
COMMENT ON VIEW public.active_drop_views_people IS 'active people';
COMMENT ON VIEW public.other_drop_views_people IS 'other people';
\dv+
DROP VIEW public.active_drop_views_people, public.missing_drop_views_people;
\dv+
DROP VIEW IF EXISTS public.active_drop_views_people, public.missing_drop_views_people;
\dv+
SELECT schemaname, viewname, viewowner, definition
FROM pg_catalog.pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','v')
ORDER BY c.relname, d.objsubid;
\dd public.*drop_views*
DROP VIEW public.other_drop_views_people, public.third_drop_views_people;
\dv+
SELECT * FROM drop_views_people ORDER BY id;
CREATE VIEW public.duplicate_drop_views_people AS SELECT id, name FROM drop_views_people ORDER BY id;
DROP VIEW public.duplicate_drop_views_people, public.duplicate_drop_views_people;
\dv+
DROP VIEW public.drop_views_people, public.duplicate_drop_views_people;
\dv+
DROP VIEW public.duplicate_drop_views_people CASCADE;
DROP VIEW private.duplicate_drop_views_people;
SELECT * FROM duplicate_drop_views_people;
