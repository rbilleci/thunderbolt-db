\echo === psql bounded sequence-backed defaults ===
CREATE TABLE serial_people (id SERIAL PRIMARY KEY, name TEXT);
\d serial_people
\ds
INSERT INTO serial_people (name) VALUES ('Ada'), ('Linus');
SELECT id, name FROM serial_people ORDER BY id;
SELECT last_value, is_called FROM public.serial_people_id_seq;
SELECT currval('public.serial_people_id_seq'::regclass);
CREATE SEQUENCE public.manual_people_seq;
CREATE TABLE manual_people (
  id INT DEFAULT nextval('public.manual_people_seq'::regclass),
  name TEXT
);
INSERT INTO manual_people (name) VALUES ('Grace'), ('Barbara');
SELECT id, name FROM manual_people ORDER BY id;
SELECT last_value, is_called FROM public.manual_people_seq;
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
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'serial_people'
ORDER BY ordinal_position;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'manual_people'
ORDER BY ordinal_position;
ALTER TABLE ONLY public.manual_people ALTER COLUMN id DROP DEFAULT;
ALTER TABLE ONLY public.serial_people ALTER COLUMN id DROP DEFAULT;
INSERT INTO manual_people (name) VALUES ('No default');
INSERT INTO serial_people (name) VALUES ('No serial default');
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
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'manual_people'
ORDER BY ordinal_position;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'serial_people'
ORDER BY ordinal_position;
\d manual_people
\d serial_people
SELECT last_value, is_called FROM public.manual_people_seq;
SELECT last_value, is_called FROM public.serial_people_id_seq;
CREATE TABLE missing_default (
  id INT DEFAULT nextval('missing_seq'::regclass),
  name TEXT
);
CREATE TABLE table_default_target (id INT);
CREATE TABLE bad_default (
  id INT DEFAULT nextval('table_default_target'::regclass),
  name TEXT
);
