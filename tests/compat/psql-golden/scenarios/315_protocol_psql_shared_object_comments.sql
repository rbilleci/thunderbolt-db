\pset pager off
\echo === shared object comments ===
COMMENT ON ROLE postgres IS 'bootstrap role';
COMMENT ON TABLESPACE pg_default IS 'default storage';
COMMENT ON TABLESPACE pg_global IS 'global storage';
\du+
\db+
SELECT pg_catalog.shobj_description(10, 'pg_authid') AS role_comment;
SELECT pg_catalog.shobj_description(1663, 'pg_tablespace') AS tablespace_comment;
SELECT pg_catalog.shobj_description(1664, 'pg_tablespace') AS global_tablespace_comment;
COMMENT ON ROLE postgres IS NULL;
COMMENT ON TABLESPACE pg_default IS NULL;
COMMENT ON TABLESPACE pg_global IS NULL;
SELECT pg_catalog.shobj_description(10, 'pg_authid') IS NULL AS role_comment_cleared;
SELECT pg_catalog.shobj_description(1663, 'pg_tablespace') IS NULL AS tablespace_comment_cleared;
SELECT pg_catalog.shobj_description(1664, 'pg_tablespace') IS NULL AS global_tablespace_comment_cleared;
COMMENT ON ROLE missing_role IS 'bad';
COMMENT ON TABLESPACE missing_space IS 'bad';
\du
