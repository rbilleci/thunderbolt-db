\echo === pg_catalog empty column default metadata ===
CREATE TABLE pg_attrdefs_people (id INT, name TEXT);
CREATE TABLE pg_attrdefs_teams (team_id INT);
SELECT n.nspname,
       c.relname,
       a.attname,
       pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS default_expr
FROM pg_catalog.pg_attrdef d
JOIN pg_catalog.pg_class c ON c.oid = d.adrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
WHERE n.nspname = 'public'
ORDER BY c.relname, a.attnum;
