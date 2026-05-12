\echo === psql dt schema qualified table pattern meta-command ===
CREATE TABLE dt_pattern_people (id INT, name TEXT);
CREATE TABLE dt_pattern_teams (id INT);
CREATE TABLE other_pattern_people (id INT);
\dt public.dt_pattern_*
