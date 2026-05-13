\echo === information_schema empty view metadata ===
CREATE TABLE is_views_people (id INT, name TEXT);
SELECT table_catalog,
       table_schema,
       table_name,
       view_definition,
       check_option,
       is_updatable,
       is_insertable_into,
       is_trigger_updatable,
       is_trigger_deletable,
       is_trigger_insertable
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;
