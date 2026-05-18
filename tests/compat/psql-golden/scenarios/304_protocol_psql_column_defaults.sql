\echo === literal column defaults ===
CREATE TABLE default_people (id INT, name TEXT DEFAULT 'unknown'::text, bucket INT DEFAULT 7);
INSERT INTO default_people (id) VALUES (1);
ALTER TABLE ONLY public.default_people ALTER COLUMN name SET DEFAULT 'changed'::text;
INSERT INTO default_people (id) VALUES (2);
SELECT id, name, bucket FROM default_people ORDER BY id;
SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'default_people' ORDER BY ordinal_position;
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
\d default_people
