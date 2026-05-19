\echo === bounded rename table ===
CREATE TABLE rename_table_people (id INT PRIMARY KEY, name TEXT UNIQUE, bucket INT DEFAULT 7);
INSERT INTO rename_table_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COMMENT ON TABLE public.rename_table_people IS 'old table';
COMMENT ON COLUMN public.rename_table_people.name IS 'person name';
COMMENT ON CONSTRAINT rename_table_people_pkey ON public.rename_table_people IS 'primary id';
\dt+ public.rename_table*
\d+ public.rename_table_people
ALTER TABLE ONLY public.rename_table_people RENAME TO renamed_table_people;
\dt+ public.rename_table*
\dt+ public.renamed_table_people
\d+ public.renamed_table_people
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'renamed_table_people'
ORDER BY ordinal_position;
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
SELECT n.nspname, c.relname, c.relkind, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public'
  AND c.relkind IN ('r','i')
ORDER BY c.relkind, c.relname, d.objsubid;
\dd rename_table_people_pkey
\di+
INSERT INTO renamed_table_people (id, name) VALUES (3, 'Grace');
SELECT id, name, bucket FROM renamed_table_people WHERE id = 2;
SELECT id FROM rename_table_people WHERE id = 1;
ALTER TABLE renamed_table_people RENAME TO renamed_table_people;
ALTER TABLE missing_rename_table_people RENAME TO renamed_missing_people;
ALTER TABLE IF EXISTS missing_rename_table_people RENAME TO renamed_missing_people;
CREATE VIEW rename_table_view AS SELECT id, name FROM renamed_table_people;
ALTER TABLE renamed_table_people RENAME TO renamed_table_blocked_people;
ALTER TABLE rename_table_view RENAME TO renamed_view;
ALTER TABLE renamed_table_people RENAME TO public.renamed_table_schema_people;
SELECT id, name, bucket FROM renamed_table_people WHERE name = 'Grace';
