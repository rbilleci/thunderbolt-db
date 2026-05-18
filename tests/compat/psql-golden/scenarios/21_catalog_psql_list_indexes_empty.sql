\echo === psql di index meta-command supported subset ===
CREATE TABLE di_empty_people (id INT, name TEXT);
CREATE INDEX di_empty_people_name_idx ON di_empty_people (name);
\di
\di public.*
