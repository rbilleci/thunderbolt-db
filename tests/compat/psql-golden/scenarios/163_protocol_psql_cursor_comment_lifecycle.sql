\echo === psql cursor comment lifecycle ===
CREATE TABLE cursor_comment_people (id INT, name TEXT);
INSERT INTO cursor_comment_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE "comment cursor" CURSOR FOR SELECT id, name FROM cursor_comment_people ORDER BY id;
FETCH /* direction */ NEXT /* marker */ FROM /* target */ "comment cursor";
MOVE /* count */ 1 /* marker */ FROM /* target */ "comment cursor";
FETCH /* remaining */ ALL /* marker */ FROM /* target */ "comment cursor";
CLOSE /* target */ "comment cursor";
FETCH /* after-close */ NEXT FROM "comment cursor";
SELECT name FROM cursor_comment_people WHERE id = 2;
