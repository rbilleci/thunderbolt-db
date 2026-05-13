\echo === pg_catalog empty view metadata ===
CREATE TABLE pg_views_people (id INT, name TEXT);
SELECT schemaname,
       viewname,
       viewowner,
       definition
FROM pg_catalog.pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
