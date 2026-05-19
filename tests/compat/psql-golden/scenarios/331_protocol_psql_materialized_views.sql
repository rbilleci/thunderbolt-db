\echo === psql bounded materialized view lifecycle ===
CREATE TABLE matview_people (id INT, name TEXT);
INSERT INTO matview_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
CREATE MATERIALIZED VIEW public.mv_people AS SELECT id, name FROM matview_people WHERE id > 1 ORDER BY id;
COMMENT ON MATERIALIZED VIEW public.mv_people IS 'people snapshot';
\dm
\dm+
SELECT * FROM mv_people;
INSERT INTO matview_people (id, name) VALUES (4, 'Barbara');
SELECT * FROM mv_people;
REFRESH MATERIALIZED VIEW public.mv_people;
SELECT * FROM mv_people;
SELECT c.oid, n.nspname, c.relname, c.relkind, c.relpersistence
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'm'
ORDER BY c.relname;
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','v','m','s')
ORDER BY c.relname, d.objsubid;
\dd mv_*
ALTER MATERIALIZED VIEW public.mv_people RENAME TO mv_people_snapshot;
\dm+
SELECT * FROM mv_people_snapshot;
SELECT * FROM mv_people;
CREATE MATERIALIZED VIEW public.mv_other AS SELECT id, name FROM matview_people WHERE id = 1;
ALTER MATERIALIZED VIEW mv_people_snapshot RENAME TO mv_other;
ALTER MATERIALIZED VIEW missing_mv RENAME TO mv_missing;
ALTER MATERIALIZED VIEW matview_people RENAME TO mv_table_target;
ALTER MATERIALIZED VIEW mv_people_snapshot RENAME TO public.mv_schema_target;
REFRESH MATERIALIZED VIEW missing_mv;
REFRESH MATERIALIZED VIEW matview_people;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_people_snapshot;
REFRESH MATERIALIZED VIEW mv_people_snapshot WITH NO DATA;
DROP MATERIALIZED VIEW missing_mv;
DROP MATERIALIZED VIEW mv_people_snapshot, missing_mv;
\dm
DROP MATERIALIZED VIEW IF EXISTS missing_mv, mv_people_snapshot;
\dm
SELECT * FROM matview_people ORDER BY id;
DROP MATERIALIZED VIEW mv_other;
\dm
CREATE VIEW public.mv_plain_view AS SELECT id, name FROM matview_people;
CREATE MATERIALIZED VIEW mv_bad_from_view AS SELECT * FROM mv_plain_view;
CREATE TABLE matview_source (id INT);
INSERT INTO matview_source (id) VALUES (10), (11);
CREATE MATERIALIZED VIEW public.matview_people AS SELECT id FROM matview_source;
CREATE MATERIALIZED VIEW mv_with_data AS SELECT id FROM matview_source WITH DATA;
SELECT * FROM mv_with_data;
CREATE MATERIALIZED VIEW mv_without_data AS SELECT id FROM matview_source WITH NO DATA;
SELECT * FROM mv_without_data;
REFRESH MATERIALIZED VIEW mv_without_data WITH DATA;
SELECT * FROM mv_without_data;
DROP MATERIALIZED VIEW mv_with_data, mv_without_data;
DROP MATERIALIZED VIEW matview_people;
