\echo === psql describe schema-qualified prefix wildcard ===
CREATE TABLE schema_prefix_people (id INT, name TEXT);
CREATE TABLE schema_prefix_teams (team_id INT);
CREATE TABLE other_schema_prefix_people (id INT);
\d public.schema_prefix_*
\d+ public.schema_prefix_*
