\echo === psql z schema qualified privileges pattern meta-command ===
CREATE TABLE z_privileges_people (id INT, name TEXT);
CREATE TABLE z_privileges_teams (id INT);
CREATE TABLE other_z_privileges (id INT);
\z public.z_privileges_*
