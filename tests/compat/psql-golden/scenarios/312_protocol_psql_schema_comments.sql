\echo === bounded schema comments ===
COMMENT ON SCHEMA public IS 'application schema';
\dn+ public
SELECT n.nspname, pg_catalog.obj_description(n.oid, 'pg_namespace') AS description
FROM pg_catalog.pg_namespace n
WHERE n.nspname = 'public'
ORDER BY n.nspname;
\dd public
COMMENT ON SCHEMA public IS NULL;
\dn+ public
SELECT n.nspname, pg_catalog.obj_description(n.oid, 'pg_namespace') AS description
FROM pg_catalog.pg_namespace n
WHERE n.nspname = 'public'
ORDER BY n.nspname;
COMMENT ON SCHEMA private IS 'bad';
CREATE TABLE schema_comment_recovered (id INT);
INSERT INTO schema_comment_recovered (id) VALUES (1);
SELECT * FROM schema_comment_recovered ORDER BY id;
