\echo === bounded add column default ===
CREATE TABLE add_column_people (id INT, name TEXT);
INSERT INTO add_column_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
ALTER TABLE ONLY public.add_column_people ADD COLUMN bucket INT DEFAULT 7;
SELECT id, name, bucket FROM add_column_people ORDER BY id;
INSERT INTO add_column_people (id, name) VALUES (3, 'Grace');
SELECT id, name, bucket FROM add_column_people ORDER BY id;
SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'add_column_people' ORDER BY ordinal_position;
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
\d add_column_people
ALTER TABLE public.add_column_people ADD COLUMN note TEXT;
ALTER TABLE public.add_column_people ADD COLUMN bucket INT DEFAULT 9;
CREATE VIEW add_column_view AS SELECT id, name FROM add_column_people ORDER BY id;
ALTER TABLE public.add_column_view ADD COLUMN extra TEXT DEFAULT 'x'::text;
SELECT id, name, bucket FROM add_column_people ORDER BY id;
