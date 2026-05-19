\echo === bounded multi-index drop ===
CREATE TABLE drop_index_people (id INT PRIMARY KEY, name TEXT, city TEXT);
INSERT INTO drop_index_people (id, name, city) VALUES (1, 'Ada', 'London'), (2, 'Linus', 'Helsinki');
CREATE INDEX drop_index_people_name_idx ON public.drop_index_people (name);
CREATE INDEX drop_index_people_city_idx ON public.drop_index_people (city);
COMMENT ON INDEX public.drop_index_people_name_idx IS 'name lookup';
COMMENT ON INDEX public.drop_index_people_city_idx IS 'city lookup';
\di+
DROP INDEX public.drop_index_people_name_idx, public.missing_drop_index_idx;
SELECT schemaname, tablename, indexname, indexdef
FROM pg_catalog.pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
DROP INDEX IF EXISTS public.drop_index_people_name_idx, public.missing_drop_index_idx;
\di+
SELECT schemaname, tablename, indexname, indexdef
FROM pg_catalog.pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
SELECT id, name FROM drop_index_people WHERE id = 2;
DROP INDEX public.drop_index_people_pkey, public.drop_index_people_city_idx;
\di+
SELECT id, name FROM drop_index_people WHERE id = 1;
DROP INDEX public.drop_index_people_city_idx CASCADE;
DROP INDEX CONCURRENTLY public.drop_index_people_city_idx;
DROP INDEX private.drop_index_people_city_idx;
SELECT id, name FROM drop_index_people WHERE id = 1;
