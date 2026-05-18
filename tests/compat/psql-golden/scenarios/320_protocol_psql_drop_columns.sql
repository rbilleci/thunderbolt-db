\echo === bounded drop column ===
CREATE TABLE drop_column_people (id INT PRIMARY KEY, name TEXT, bucket INT DEFAULT 7);
INSERT INTO drop_column_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COMMENT ON COLUMN public.drop_column_people.name IS 'drop me';
COMMENT ON COLUMN public.drop_column_people.bucket IS 'keep me';
\d+ drop_column_people
ALTER TABLE ONLY public.drop_column_people DROP COLUMN name;
SELECT id, bucket FROM drop_column_people ORDER BY id;
INSERT INTO drop_column_people (id) VALUES (3);
SELECT id, bucket FROM drop_column_people ORDER BY id;
SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'drop_column_people' ORDER BY ordinal_position;
SELECT n.nspname,
       c.relname,
       a.attname,
       pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS default_expr
FROM pg_catalog.pg_attrdef d
JOIN pg_catalog.pg_class c ON c.oid = d.adrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
WHERE n.nspname = 'public'
ORDER BY c.relname, a.attnum;
\d+ drop_column_people
ALTER TABLE public.drop_column_people DROP COLUMN missing_name;
ALTER TABLE public.drop_column_people DROP COLUMN id;
CREATE VIEW drop_column_view AS SELECT id, bucket FROM drop_column_people ORDER BY id;
ALTER TABLE public.drop_column_view DROP COLUMN bucket;
SELECT id, bucket FROM drop_column_people ORDER BY id;
