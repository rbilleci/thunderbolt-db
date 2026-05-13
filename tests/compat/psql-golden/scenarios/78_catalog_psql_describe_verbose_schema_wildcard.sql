\echo === psql verbose schema wildcard describe ===
CREATE TABLE verbose_schema_wildcard_people (id INT, name TEXT);
CREATE TABLE verbose_schema_wildcard_teams (team_id INT, team_name TEXT);
\d+ public.*
