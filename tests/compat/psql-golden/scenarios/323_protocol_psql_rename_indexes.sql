\echo === bounded rename index ===
CREATE TABLE rename_index_people (id INT, name TEXT);
INSERT INTO rename_index_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
CREATE INDEX rename_index_people_name_idx ON public.rename_index_people (name);
COMMENT ON INDEX public.rename_index_people_name_idx IS 'lookup by display name';
\di+
ALTER INDEX public.rename_index_people_name_idx RENAME TO rename_index_people_lookup_idx;
SELECT schemaname, tablename, indexname, indexdef
FROM pg_catalog.pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
SELECT n.nspname, c.relname, c.relkind, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public'
  AND c.relkind IN ('r','i')
ORDER BY c.relkind, c.relname, d.objsubid;
\di+
SELECT id, name FROM rename_index_people WHERE name = 'Linus';
ALTER INDEX public.rename_index_people_lookup_idx RENAME TO rename_index_people_lookup_idx;
ALTER INDEX public.missing_rename_index_people_idx RENAME TO rename_index_people_old_idx;
CREATE TABLE rename_index_keyed_people (id INT PRIMARY KEY);
ALTER INDEX public.rename_index_keyed_people_pkey RENAME TO rename_index_keyed_people_id_idx;
ALTER INDEX public.rename_index_people_lookup_idx RENAME TO public.rename_index_people_schema_idx;
SELECT id, name FROM rename_index_people WHERE name = 'Linus';
