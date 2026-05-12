\echo === psql dt schema qualified meta-command ===
CREATE TABLE dt_schema_people (id INT, name TEXT);
CREATE TABLE dt_schema_teams (id INT);
\dt public.*
