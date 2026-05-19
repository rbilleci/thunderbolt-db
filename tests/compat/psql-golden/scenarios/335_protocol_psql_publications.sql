\echo === protocol psql bounded publications ===
CREATE TABLE pub_people (id INT, name TEXT);
CREATE TABLE pub_accounts (id INT, owner TEXT);
CREATE PUBLICATION app_pub FOR TABLE public.pub_people, pub_accounts;
CREATE PUBLICATION all_pub FOR ALL TABLES;
\dRp
SELECT pubname, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot
FROM pg_catalog.pg_publication
ORDER BY pubname;
SELECT p.pubname, c.relname
FROM pg_catalog.pg_publication p
JOIN pg_catalog.pg_publication_rel pr ON pr.prpubid = p.oid
JOIN pg_catalog.pg_class c ON c.oid = pr.prrelid
ORDER BY p.pubname, c.relname;
\d+ pub_people
CREATE PUBLICATION app_pub FOR ALL TABLES;
CREATE PUBLICATION missing_pub FOR TABLE missing_people;
CREATE VIEW pub_people_view AS SELECT * FROM pub_people;
CREATE PUBLICATION view_pub FOR TABLE pub_people_view;
CREATE PUBLICATION bad_pub FOR TABLE pub_people WITH (publish = 'insert');
DROP PUBLICATION missing_pub;
DROP PUBLICATION IF EXISTS missing_pub;
DROP PUBLICATION app_pub;
\dRp
\d+ pub_people
DROP PUBLICATION all_pub;
\dRp
SELECT id, name FROM pub_people ORDER BY id;
