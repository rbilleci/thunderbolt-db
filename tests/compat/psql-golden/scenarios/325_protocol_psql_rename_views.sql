\echo === bounded SQL view rename ===
CREATE TABLE rename_view_people (id INT, name TEXT);
INSERT INTO rename_view_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE VIEW public.active_rename_view_people AS SELECT id, name FROM rename_view_people WHERE id > 1 ORDER BY id;
COMMENT ON VIEW public.active_rename_view_people IS 'active renamed view';
ALTER VIEW public.active_rename_view_people RENAME TO renamed_view_people;
SELECT * FROM renamed_view_people;
SELECT schemaname, viewname, viewowner, definition
FROM pg_catalog.pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
SELECT table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','v')
ORDER BY c.relname, d.objsubid;
\dv+
\dd
SELECT * FROM active_rename_view_people;
CREATE VIEW public.other_rename_view_people AS SELECT id, name FROM rename_view_people WHERE id = 1;
ALTER VIEW renamed_view_people RENAME TO other_rename_view_people;
ALTER VIEW missing_rename_view_people RENAME TO still_missing_view;
ALTER VIEW rename_view_people RENAME TO table_target_view;
ALTER MATERIALIZED VIEW renamed_view_people RENAME TO mat_view_target;
ALTER VIEW renamed_view_people RENAME TO public.schema_qualified_view;
ALTER VIEW renamed_view_people RENAME TO cascade_view CASCADE;
SELECT * FROM renamed_view_people;
SELECT * FROM rename_view_people ORDER BY id;
