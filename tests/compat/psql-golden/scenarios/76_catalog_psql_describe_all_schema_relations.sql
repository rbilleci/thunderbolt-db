\echo === psql all-schema relation describe ===
CREATE TABLE all_schema_describe_people (id INT, name TEXT);
CREATE TABLE all_schema_describe_teams (team_id INT, team_name TEXT);
\d *.*
