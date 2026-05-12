\echo === pg_catalog empty description metadata ===
CREATE TABLE pg_descriptions_people (id INT, name TEXT);
CREATE TABLE pg_descriptions_teams (team_id INT);
SELECT n.nspname,
       c.relname,
       a.attname,
       d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
ORDER BY c.relname, d.objsubid;
