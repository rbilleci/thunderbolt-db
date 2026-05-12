\echo === psql d schema qualified table meta-command ===
CREATE TABLE describe_schema_people (id INT, name TEXT);
\d public.describe_schema_people
\echo === psql d plus schema qualified table meta-command ===
\d+ public.describe_schema_people
