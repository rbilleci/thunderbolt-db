\echo === psql drop table clean-restore boundary ===
CREATE TABLE drop_clean_people (id int4, name text);
INSERT INTO drop_clean_people (id, name) VALUES (99, 'stale');
DROP TABLE IF EXISTS public.drop_clean_people;
CREATE TABLE drop_clean_people (id int4, name text);
INSERT INTO drop_clean_people (id, name) VALUES (1, 'Ada');
SELECT id, name FROM drop_clean_people ORDER BY id;
DROP TABLE IF EXISTS public.drop_clean_missing;
DROP SCHEMA IF EXISTS public;
CREATE SCHEMA public;
SELECT name FROM drop_clean_people WHERE id = 1;
