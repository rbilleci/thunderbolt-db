\echo === bounded SQL multi-table drops ===
CREATE TABLE drop_table_batch_people (id INT PRIMARY KEY, name TEXT);
CREATE TABLE drop_table_batch_teams (id INT, name TEXT UNIQUE);
CREATE TABLE drop_table_batch_keep (id INT, name TEXT);
INSERT INTO drop_table_batch_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
INSERT INTO drop_table_batch_teams (id, name) VALUES (10, 'Compiler');
CREATE INDEX drop_table_batch_people_name_idx ON drop_table_batch_people (name);
COMMENT ON TABLE public.drop_table_batch_people IS 'people table';
COMMENT ON COLUMN public.drop_table_batch_teams.name IS 'team name';
COMMENT ON INDEX public.drop_table_batch_people_name_idx IS 'lookup';
\dt drop_table_batch*
\di+
DROP TABLE public.drop_table_batch_people, public.drop_table_batch_teams;
\dt drop_table_batch*
\di+
\dd drop_table_batch*
SELECT id, name FROM drop_table_batch_keep ORDER BY id;
DROP TABLE drop_table_batch_keep, missing_drop_table_batch;
\dt drop_table_batch*
DROP TABLE drop_table_batch_keep, drop_table_batch_keep;
\dt drop_table_batch*
CREATE TABLE drop_table_batch_base (id INT, name TEXT);
CREATE VIEW public.drop_table_batch_view AS SELECT id, name FROM drop_table_batch_base;
DROP TABLE IF EXISTS missing_drop_table_batch, drop_table_batch_view;
\dv+
DROP TABLE IF EXISTS missing_drop_table_batch, drop_table_batch_keep;
\dt drop_table_batch*
