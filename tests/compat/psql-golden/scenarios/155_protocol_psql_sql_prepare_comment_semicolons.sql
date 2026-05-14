\echo === psql SQL prepare comment semicolons ===
CREATE TABLE sql_prepare_comment_semicolon_people (id INT, name TEXT);
INSERT INTO sql_prepare_comment_semicolon_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
/* comment ; outer /* nested ; */ done */ PREPARE comment_semicolon_lookup(int4, text) AS SELECT id, name FROM sql_prepare_comment_semicolon_people WHERE id = $1 AND name = $2;
-- comment ; before execute
EXECUTE comment_semicolon_lookup(2, 'Linus');
DEALLOCATE PREPARE comment_semicolon_lookup;
SELECT name FROM sql_prepare_comment_semicolon_people WHERE id = 3;
