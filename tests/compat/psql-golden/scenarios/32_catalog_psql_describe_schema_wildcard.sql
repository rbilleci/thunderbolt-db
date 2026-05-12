\echo === psql describe schema wildcard ===
CREATE TABLE wildcard_people (id INT, name TEXT);
CREATE TABLE wildcard_teams (team_id INT);
\d public.*
