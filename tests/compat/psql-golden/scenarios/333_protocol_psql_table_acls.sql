\echo === bounded table ACL metadata ===
CREATE TABLE acl_people (id INT, name TEXT);
CREATE VIEW public.acl_people_view AS SELECT * FROM acl_people;
CREATE SEQUENCE public.acl_people_seq;
GRANT SELECT, INSERT ON TABLE public.acl_people TO PUBLIC;
\dp acl_people
GRANT ALL PRIVILEGES ON acl_people TO postgres;
\z public.acl_*
REVOKE INSERT ON TABLE acl_people FROM PUBLIC;
\dp acl_people
REVOKE ALL PRIVILEGES ON acl_people FROM postgres;
\z public.acl_*
GRANT SELECT ON TABLE missing_acl_people TO PUBLIC;
GRANT SELECT ON acl_people_view TO PUBLIC;
GRANT SELECT ON acl_people_seq TO PUBLIC;
GRANT UPDATE (name) ON acl_people TO PUBLIC;
GRANT SELECT ON private.acl_people TO PUBLIC;
GRANT SELECT ON acl_people TO missing_role;
GRANT SELECT ON acl_people TO PUBLIC WITH GRANT OPTION;
SELECT name FROM acl_people WHERE id = 1;
