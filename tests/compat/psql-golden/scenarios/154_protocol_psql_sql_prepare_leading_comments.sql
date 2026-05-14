\echo === psql SQL prepare leading comments ===
CREATE TABLE sql_prepare_leading_comment_people (id INT, name TEXT);
INSERT INTO sql_prepare_leading_comment_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
/* prepare prefix */ -- keyword follows
PREPARE leading_comment_lookup(int4, text) AS SELECT id, name FROM sql_prepare_leading_comment_people WHERE id = $1 AND name = $2;
-- execute prefix
EXECUTE leading_comment_lookup(3, 'Grace');
/* deallocate prefix */
DEALLOCATE PREPARE leading_comment_lookup;
/* missing execute prefix */ EXECUTE leading_comment_lookup(3, 'Grace');
SELECT name FROM sql_prepare_leading_comment_people WHERE id = 1;
