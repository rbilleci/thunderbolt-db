\echo === table and column comments ===
CREATE TABLE comment_people (id INT, name TEXT);
CREATE INDEX comment_people_name_idx ON comment_people (name);
COMMENT ON TABLE public.comment_people IS 'people lookup table';
COMMENT ON COLUMN public.comment_people.name IS 'display name';
COMMENT ON INDEX public.comment_people_name_idx IS 'name lookup index';
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY c.relname, d.objsubid;
SELECT n.nspname, c.relname, c.relkind, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','i')
ORDER BY c.relkind, c.relname, d.objsubid;
\d+ comment_people
\di+
\dd comment_people
COMMENT ON COLUMN public.comment_people.name IS NULL;
COMMENT ON INDEX public.comment_people_name_idx IS NULL;
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY c.relname, d.objsubid;
COMMENT ON TABLE public.missing_comment_people IS 'bad';
COMMENT ON COLUMN public.comment_people.missing IS 'bad';
COMMENT ON INDEX public.comment_people_idx IS 'bad';
