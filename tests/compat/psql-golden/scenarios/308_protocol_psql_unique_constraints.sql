\echo === unique constraints ===
CREATE TABLE uq_people (id INT, name TEXT UNIQUE, CONSTRAINT uq_people_id_key UNIQUE (id));
INSERT INTO uq_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT table_schema, table_name, column_name, constraint_name, ordinal_position FROM information_schema.key_column_usage WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
SELECT n.nspname, c.relname, con.conname, con.contype
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
SELECT schemaname, tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;
COMMENT ON CONSTRAINT uq_people_name_key ON public.uq_people IS 'unique display name';
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\dd uq_people_name_key
\d uq_people
INSERT INTO uq_people (id, name) VALUES (3, 'Ada');
SELECT id, name FROM uq_people ORDER BY id;
UPDATE uq_people SET id = 1 WHERE name = 'Grace';
SELECT id, name FROM uq_people ORDER BY id;
COPY uq_people FROM STDIN WITH CSV;
4,Ada
\.
SELECT id, name FROM uq_people ORDER BY id;
CREATE TABLE uq_teams (id INT, name TEXT);
INSERT INTO uq_teams (id, name) VALUES (1, 'core'), (2, 'db');
ALTER TABLE ONLY public.uq_teams ADD CONSTRAINT uq_teams_name_key UNIQUE (name);
INSERT INTO uq_teams (id, name) VALUES (3, 'core');
SELECT id, name FROM uq_teams ORDER BY id;
CREATE TABLE uq_dupes (id INT, name TEXT);
INSERT INTO uq_dupes (id, name) VALUES (1, 'a'), (2, 'a');
ALTER TABLE ONLY public.uq_dupes ADD CONSTRAINT uq_dupes_name_key UNIQUE (name);
