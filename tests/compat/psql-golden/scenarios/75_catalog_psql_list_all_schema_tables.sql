\echo === psql all-schema table listing ===
CREATE TABLE all_schema_people (id INT, name TEXT);
CREATE TABLE all_schema_teams (team_id INT);
\dt *.*
\echo === psql all-schema verbose table listing ===
\dt+ *.*
