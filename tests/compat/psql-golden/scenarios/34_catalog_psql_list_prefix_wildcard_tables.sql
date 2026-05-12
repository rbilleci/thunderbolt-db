\echo === psql list prefix wildcard tables ===
CREATE TABLE prefix_people (id INT, name TEXT);
CREATE TABLE prefix_teams (team_id INT);
CREATE TABLE other_people (id INT);
\dt prefix*
\dt+ prefix*
