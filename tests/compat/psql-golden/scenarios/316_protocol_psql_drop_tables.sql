\echo === bounded SQL table drops ===
CREATE TABLE drop_table_people (id INT PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE drop_table_teams (id INT, name TEXT);
INSERT INTO drop_table_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
CREATE INDEX drop_table_people_name_idx ON drop_table_people (name);
COMMENT ON TABLE public.drop_table_people IS 'people table';
COMMENT ON COLUMN public.drop_table_people.name IS 'display name';
COMMENT ON INDEX public.drop_table_people_name_idx IS 'lookup index';
COMMENT ON CONSTRAINT drop_table_people_pkey ON public.drop_table_people IS 'identity';
\dt drop_table*
\d+ drop_table_people
DROP TABLE public.drop_table_people;
\dt drop_table*
\dd drop_table_people
SELECT id, name FROM drop_table_people ORDER BY id;
DROP TABLE public.drop_table_people;
DROP TABLE IF EXISTS public.drop_table_people;
CREATE VIEW public.drop_table_view AS SELECT * FROM drop_table_teams;
DROP TABLE public.drop_table_view;
SELECT * FROM drop_table_view;
DROP TABLE drop_table_teams CASCADE;
DROP TABLE drop_table_teams, drop_table_missing;
SELECT id, name FROM drop_table_teams ORDER BY id;
