\echo === bounded SQL table truncates ===
CREATE TABLE truncate_people (id INT PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE truncate_teams (id INT, name TEXT);
INSERT INTO truncate_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
INSERT INTO truncate_teams (id, name) VALUES (9, 'Infra');
CREATE INDEX truncate_people_name_idx ON truncate_people (name);
COMMENT ON TABLE public.truncate_people IS 'people table';
COMMENT ON COLUMN public.truncate_people.name IS 'display name';
COMMENT ON INDEX public.truncate_people_name_idx IS 'lookup index';
COMMENT ON CONSTRAINT truncate_people_pkey ON public.truncate_people IS 'identity';
\d+ truncate_people
TRUNCATE TABLE ONLY public.truncate_people;
SELECT id, name FROM truncate_people ORDER BY id;
\d+ truncate_people
\dd truncate_people
INSERT INTO truncate_people (id, name) VALUES (1, 'Ada');
SELECT id, name FROM truncate_people ORDER BY id;
SELECT id, name FROM truncate_teams ORDER BY id;
CREATE VIEW public.truncate_view AS SELECT * FROM truncate_teams;
TRUNCATE truncate_view;
SELECT * FROM truncate_view;
TRUNCATE TABLE missing_truncate;
TRUNCATE TABLE truncate_teams RESTART IDENTITY;
TRUNCATE TABLE truncate_teams CASCADE;
TRUNCATE TABLE truncate_teams, truncate_people;
