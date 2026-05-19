\echo === protocol psql bounded subscriptions ===
CREATE TABLE sub_people (id INT, name TEXT);
CREATE PUBLICATION sub_pub FOR TABLE sub_people;
CREATE PUBLICATION sub_all FOR ALL TABLES;
CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost dbname=postgres' PUBLICATION sub_pub, sub_all WITH (connect = false, enabled = false);
COMMENT ON PUBLICATION sub_pub IS 'subscription source publication';
COMMENT ON SUBSCRIPTION app_sub IS 'disabled subscription metadata';
\dRs
\dd
SELECT description, classoid, objoid, objsubid
FROM pg_catalog.pg_description
ORDER BY classoid, objoid, objsubid;
SELECT subname, subenabled, subconninfo, subpublications
FROM pg_catalog.pg_subscription
ORDER BY subname;
COMMENT ON PUBLICATION missing_pub IS 'missing';
COMMENT ON SUBSCRIPTION missing_sub IS 'missing';
COMMENT ON SUBSCRIPTION app_sub IS NULL;
SELECT description, classoid, objoid, objsubid
FROM pg_catalog.pg_description
ORDER BY classoid, objoid, objsubid;
CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION sub_pub WITH (connect = false, enabled = false);
CREATE SUBSCRIPTION missing_pub_sub CONNECTION 'host=localhost' PUBLICATION missing_pub WITH (connect = false, enabled = false);
CREATE SUBSCRIPTION duplicate_pub_sub CONNECTION 'host=localhost' PUBLICATION sub_pub, sub_pub WITH (connect = false, enabled = false);
CREATE SUBSCRIPTION enabled_sub CONNECTION 'host=localhost' PUBLICATION sub_pub WITH (connect = false, enabled = true);
CREATE SUBSCRIPTION connect_sub CONNECTION 'host=localhost' PUBLICATION sub_pub WITH (connect = true, enabled = false);
CREATE SUBSCRIPTION implicit_streaming_sub CONNECTION 'host=localhost' PUBLICATION sub_pub;
DROP SUBSCRIPTION missing_sub;
DROP SUBSCRIPTION IF EXISTS missing_sub;
DROP SUBSCRIPTION app_sub;
\dRs
DROP PUBLICATION sub_pub, sub_all;
\dRp
SELECT description, classoid, objoid, objsubid
FROM pg_catalog.pg_description
ORDER BY classoid, objoid, objsubid;
