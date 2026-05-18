\echo === bounded rename column ===
CREATE TABLE rename_column_people (id INT PRIMARY KEY, name TEXT DEFAULT 'unknown');
INSERT INTO rename_column_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COMMENT ON COLUMN public.rename_column_people.name IS 'display name';
\d+ rename_column_people
ALTER TABLE ONLY public.rename_column_people RENAME COLUMN name TO display_name;
ALTER TABLE public.rename_column_people RENAME COLUMN id TO person_id;
INSERT INTO rename_column_people (person_id) VALUES (3);
SELECT person_id, display_name FROM rename_column_people ORDER BY person_id;
SELECT person_id, display_name FROM rename_column_people WHERE person_id = 2;
SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'rename_column_people' ORDER BY ordinal_position;
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
\d+ rename_column_people
ALTER TABLE public.rename_column_people RENAME COLUMN display_name TO person_id;
ALTER TABLE public.rename_column_people RENAME COLUMN missing_name TO nickname;
CREATE VIEW rename_column_view AS SELECT person_id, display_name FROM rename_column_people ORDER BY person_id;
ALTER TABLE public.rename_column_view RENAME COLUMN display_name TO nickname;
SELECT person_id, display_name FROM rename_column_people ORDER BY person_id;
