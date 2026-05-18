\echo === bounded SQL view comments ===
CREATE TABLE comment_view_people (id INT, name TEXT);
INSERT INTO comment_view_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
CREATE VIEW public.active_comment_view_people AS SELECT id, name FROM comment_view_people WHERE id > 1 ORDER BY id;
COMMENT ON VIEW public.active_comment_view_people IS 'active people view';
\dv+
SELECT n.nspname, c.relname, a.attname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_class c ON c.oid = d.objoid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
WHERE n.nspname = 'public' AND c.relkind IN ('r','v')
ORDER BY c.relname, d.objsubid;
\dd
COMMENT ON VIEW public.active_comment_view_people IS NULL;
\dv+
COMMENT ON VIEW public.missing_view IS 'bad';
COMMENT ON VIEW public.comment_view_people IS 'bad';
COMMENT ON MATERIALIZED VIEW public.active_comment_view_people IS 'bad';
DROP VIEW public.active_comment_view_people;
COMMENT ON VIEW public.active_comment_view_people IS 'bad';
SELECT * FROM comment_view_people ORDER BY id;
