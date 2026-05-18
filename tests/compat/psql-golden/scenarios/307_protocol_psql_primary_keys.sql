\echo === primary key constraints ===
CREATE TABLE pk_people (id INT PRIMARY KEY, name TEXT);
INSERT INTO pk_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT table_schema, table_name, column_name, constraint_name, ordinal_position FROM information_schema.key_column_usage WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
SELECT n.nspname, c.relname, con.conname, con.contype
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
SELECT schemaname, tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;
COMMENT ON CONSTRAINT pk_people_pkey ON public.pk_people IS 'row identity';
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\dd pk_people_pkey
COMMENT ON CONSTRAINT pk_people_pkey ON public.pk_people IS NULL;
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\d pk_people
INSERT INTO pk_people (id, name) VALUES (1, 'Edsger');
SELECT id, name FROM pk_people ORDER BY id;
UPDATE pk_people SET id = 1 WHERE name = 'Grace';
SELECT id, name FROM pk_people ORDER BY id;
CREATE TABLE pk_teams (id INT, name TEXT);
INSERT INTO pk_teams (id, name) VALUES (1, 'core'), (2, 'db');
ALTER TABLE ONLY public.pk_teams ADD CONSTRAINT pk_teams_pkey PRIMARY KEY (id);
INSERT INTO pk_teams (id, name) VALUES (1, 'dup');
SELECT id, name FROM pk_teams ORDER BY id;
CREATE TABLE pk_dupes (id INT, name TEXT);
INSERT INTO pk_dupes (id, name) VALUES (1, 'a'), (1, 'b');
ALTER TABLE ONLY public.pk_dupes ADD CONSTRAINT pk_dupes_pkey PRIMARY KEY (id);
