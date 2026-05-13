\echo === psql verbose schema-wildcard table listing ===
CREATE TABLE verbose_schema_list_people (id INT, name TEXT);
CREATE TABLE verbose_schema_list_teams (team_id INT);
\dt+ public.*
