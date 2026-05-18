\echo === bounded SQL create or replace views ===
CREATE TABLE replace_view_people (id INT, name TEXT);
INSERT INTO replace_view_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE VIEW public.active_replace_view_people AS SELECT id, name FROM replace_view_people WHERE id > 1 ORDER BY id;
COMMENT ON VIEW public.active_replace_view_people IS 'active replacement view';
CREATE OR REPLACE VIEW public.active_replace_view_people AS SELECT id, name FROM replace_view_people WHERE id > 2 ORDER BY id;
SELECT * FROM active_replace_view_people;
SELECT schemaname, viewname, viewowner, definition
FROM pg_catalog.pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
SELECT table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;
\dv+
CREATE OR REPLACE VIEW public.active_replace_view_people AS SELECT * FROM missing_replace_view_people;
SELECT * FROM active_replace_view_people;
CREATE OR REPLACE VIEW public.replace_view_people AS SELECT * FROM replace_view_people;
SELECT * FROM replace_view_people ORDER BY id;
