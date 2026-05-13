\echo === psql schema-qualified exact table listing ===
CREATE TABLE exact_schema_list_people (id INT, name TEXT);
CREATE TABLE exact_schema_list_teams (id INT);
\dt public.exact_schema_list_people
