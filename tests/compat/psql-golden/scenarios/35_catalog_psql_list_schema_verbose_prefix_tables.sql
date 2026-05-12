\echo === psql list schema-qualified verbose prefix tables ===
CREATE TABLE schema_verbose_people (id INT, name TEXT);
CREATE TABLE schema_verbose_teams (team_id INT);
CREATE TABLE other_verbose_people (id INT);
\dt+ public.schema_verbose_*
\dt+ public.schema_verbose_people
