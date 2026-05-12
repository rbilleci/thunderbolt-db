\echo === psql describe prefix wildcard ===
CREATE TABLE prefix_people (id INT, name TEXT);
CREATE TABLE prefix_teams (team_id INT);
\d prefix_*
\d+ prefix_*
