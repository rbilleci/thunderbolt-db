\echo === bounded rename constraint ===
CREATE TABLE rename_constraint_people (id INT PRIMARY KEY, name TEXT UNIQUE);
INSERT INTO rename_constraint_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COMMENT ON CONSTRAINT rename_constraint_people_pkey ON public.rename_constraint_people IS 'row identity';
COMMENT ON INDEX public.rename_constraint_people_name_key IS 'unique display name index';
\d+ rename_constraint_people
ALTER TABLE ONLY public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_pkey TO rename_constraint_people_id_pkey;
ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_name_key TO rename_constraint_people_display_name_key;
SELECT table_schema, table_name, constraint_name, constraint_type
FROM information_schema.table_constraints
WHERE table_schema = 'public'
ORDER BY table_name, constraint_name;
SELECT table_schema, table_name, column_name, constraint_name, ordinal_position
FROM information_schema.key_column_usage
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
SELECT n.nspname, c.relname, con.conname, con.contype
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
SELECT schemaname, tablename, indexname, indexdef
FROM pg_catalog.pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\dd rename_constraint_people_id_pkey
\di+
\d+ rename_constraint_people
INSERT INTO rename_constraint_people (id, name) VALUES (3, 'Ada');
ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_id_pkey TO rename_constraint_people_display_name_key;
ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT missing_constraint TO renamed_missing;
ALTER TABLE IF EXISTS ONLY public.missing_rename_constraint_people RENAME CONSTRAINT missing_constraint TO renamed_missing;
CREATE VIEW rename_constraint_view AS SELECT id, name FROM rename_constraint_people ORDER BY id;
ALTER TABLE public.rename_constraint_view RENAME CONSTRAINT missing_constraint TO renamed_missing;
ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_id_pkey TO rename_constraint_people_old_pkey CASCADE;
SELECT id, name FROM rename_constraint_people ORDER BY id;
