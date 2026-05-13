\echo === psql verbose all-schema relation describe ===
CREATE TABLE all_schema_verbose_people (id INT, name TEXT);
CREATE TABLE all_schema_verbose_teams (team_id INT, team_name TEXT);
\d+ *.*
