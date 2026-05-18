\echo === bounded SQL view drops ===
CREATE TABLE drop_view_people (id INT, name TEXT);
INSERT INTO drop_view_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE VIEW public.active_drop_view_people AS SELECT id, name FROM drop_view_people WHERE id > 1 ORDER BY id;
CREATE VIEW public.other_drop_view_people AS SELECT id, name FROM drop_view_people WHERE id = 1;
SELECT * FROM active_drop_view_people;
DROP VIEW public.active_drop_view_people;
SELECT schemaname, viewname, viewowner, definition
FROM pg_catalog.pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
SELECT table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;
\dv
\dv+
SELECT * FROM drop_view_people ORDER BY id;
SELECT * FROM active_drop_view_people;
DROP VIEW active_drop_view_people;
DROP VIEW IF EXISTS active_drop_view_people;
DROP VIEW drop_view_people;
DROP VIEW other_drop_view_people CASCADE;
SELECT * FROM other_drop_view_people;
