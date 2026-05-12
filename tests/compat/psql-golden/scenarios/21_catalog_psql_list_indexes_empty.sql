\echo === psql di index meta-command empty subset ===
CREATE TABLE di_empty_people (id INT, name TEXT);
\di
\di public.*
