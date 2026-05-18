\echo === bounded SQL views ===
CREATE TABLE view_people (id INT, name TEXT);
INSERT INTO view_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE VIEW public.active_view_people AS SELECT id, name FROM view_people WHERE id > 1 ORDER BY id;
SELECT * FROM active_view_people;
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
CREATE VIEW public.missing_view_people AS SELECT * FROM missing_view_people;
CREATE VIEW public.active_view_people AS SELECT * FROM view_people;
CREATE VIEW public.nested_view_people AS SELECT * FROM active_view_people;
SELECT id FROM active_view_people WHERE id = 2;
SELECT * FROM view_people ORDER BY id;
