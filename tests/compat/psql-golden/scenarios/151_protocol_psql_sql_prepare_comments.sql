\echo === psql SQL prepare comments ===
CREATE TABLE sql_prepare_comment_people (id INT, name TEXT);
INSERT INTO sql_prepare_comment_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE comment_lookup(/* id */ int4, /* name */ pg_catalog.text) /* target */ AS SELECT id, name FROM sql_prepare_comment_people WHERE id = $1 AND name = $2;
EXECUTE /* stmt */ comment_lookup(/* id */ 2, /* name */ 'Linus' /* trailing */);
EXECUTE comment_lookup(3 /* id */, 'Grace'::text /* name */);
DEALLOCATE /* scope */ PREPARE /* target */ comment_lookup;
EXECUTE comment_lookup(2, 'Linus');
SELECT name FROM sql_prepare_comment_people WHERE id = 1;
