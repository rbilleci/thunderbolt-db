\echo === psql cursor declare comments ===
CREATE TABLE cursor_declare_comment_people (id INT, name TEXT);
INSERT INTO cursor_declare_comment_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE /* cursor name */ "declare comment cursor" /* sensitivity */ ASENSITIVE /* direction */ NO SCROLL /* cursor keyword */ CURSOR /* lifetime */ WITHOUT HOLD /* query */ FOR SELECT id, name FROM cursor_declare_comment_people ORDER BY id;
FETCH NEXT FROM "declare comment cursor";
MOVE 1 FROM "declare comment cursor";
FETCH ALL FROM "declare comment cursor";
CLOSE "declare comment cursor";
FETCH NEXT FROM "declare comment cursor";
SELECT name FROM cursor_declare_comment_people WHERE id = 2;
