\echo === bounded check constraints ===
CREATE TABLE check_people (id INT, name TEXT, CONSTRAINT check_people_id_positive CHECK (id > 0));
INSERT INTO check_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
INSERT INTO check_people (id, name) VALUES (-1, 'Bad');
SELECT id, name FROM check_people ORDER BY id;
UPDATE check_people SET id = -2 WHERE name = 'Linus';
SELECT id, name FROM check_people ORDER BY id;
COMMENT ON CONSTRAINT check_people_id_positive ON public.check_people IS 'positive ids only';
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT n.nspname, c.relname, con.conname, con.contype
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\d+ check_people
ALTER TABLE ONLY public.check_people RENAME CONSTRAINT check_people_id_positive TO check_people_id_gt_zero;
\d+ check_people
ALTER TABLE ONLY public.check_people DROP CONSTRAINT check_people_id_gt_zero;
INSERT INTO check_people (id, name) VALUES (-3, 'Allowed after drop');
SELECT id, name FROM check_people ORDER BY id;
CREATE TABLE check_existing (id INT, name TEXT);
INSERT INTO check_existing (id, name) VALUES (1, 'ok'), (-1, 'bad');
ALTER TABLE ONLY public.check_existing ADD CONSTRAINT check_existing_id_positive CHECK (id > 0);
SELECT id, name FROM check_existing ORDER BY id;
ALTER TABLE ONLY public.check_existing ADD CONSTRAINT check_existing_missing CHECK (missing > 0);
CREATE TABLE check_missing_column (id INT, CONSTRAINT check_missing_column_bad CHECK (missing > 0));
CREATE TABLE check_unsupported (id INT, CONSTRAINT check_unsupported_between CHECK (id BETWEEN 1 AND 3));
