\echo === psql SQL prepare comment as separator ===
CREATE TABLE sql_prepare_comment_as_people (id INT, name TEXT);
INSERT INTO sql_prepare_comment_as_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE comment_as_lookup -- AS inside a line comment must not terminate PREPARE target
(int4, text) /* nested AS /* inner AS */ still target */ AS SELECT id, name FROM sql_prepare_comment_as_people WHERE id = $1 AND name = $2;
EXECUTE comment_as_lookup(2, 'Linus');
DEALLOCATE PREPARE comment_as_lookup;
EXECUTE comment_as_lookup(2, 'Linus');
SELECT name FROM sql_prepare_comment_as_people WHERE id = 1;
