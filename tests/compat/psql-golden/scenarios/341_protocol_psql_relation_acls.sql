\echo === bounded relation ACL metadata ===
CREATE TABLE rel_acl_people (id INT, name TEXT);
INSERT INTO rel_acl_people VALUES (1, 'ada'), (2, 'grace');
CREATE VIEW public.rel_acl_view AS SELECT * FROM rel_acl_people;
CREATE MATERIALIZED VIEW public.rel_acl_mv AS SELECT * FROM rel_acl_people WITH DATA;
CREATE SEQUENCE public.rel_acl_seq;
GRANT SELECT ON VIEW public.rel_acl_view TO PUBLIC;
GRANT SELECT ON MATERIALIZED VIEW public.rel_acl_mv TO PUBLIC;
GRANT SELECT, UPDATE ON SEQUENCE public.rel_acl_seq TO postgres;
\dp public.rel_acl_*
\z rel_acl_*
ALTER VIEW public.rel_acl_view RENAME TO rel_acl_view_renamed;
ALTER MATERIALIZED VIEW public.rel_acl_mv RENAME TO rel_acl_mv_renamed;
ALTER SEQUENCE public.rel_acl_seq RENAME TO rel_acl_seq_renamed;
\dp public.rel_acl_*
REVOKE SELECT ON VIEW rel_acl_view_renamed FROM PUBLIC;
REVOKE UPDATE ON SEQUENCE rel_acl_seq_renamed FROM postgres;
\z public.rel_acl_*
DROP VIEW rel_acl_view_renamed;
DROP MATERIALIZED VIEW rel_acl_mv_renamed;
DROP SEQUENCE rel_acl_seq_renamed;
\dp public.rel_acl_*
GRANT SELECT ON VIEW rel_acl_people TO PUBLIC;
GRANT SELECT ON MATERIALIZED VIEW rel_acl_people TO PUBLIC;
GRANT SELECT ON SEQUENCE rel_acl_people TO PUBLIC;
GRANT SELECT ON VIEW missing_rel_acl_view TO PUBLIC;
SELECT name FROM rel_acl_people WHERE id = 2;
