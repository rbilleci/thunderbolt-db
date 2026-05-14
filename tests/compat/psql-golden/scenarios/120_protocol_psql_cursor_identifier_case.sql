\echo === psql cursor identifier case lifecycle ===
CREATE TABLE cursor_case_people (id INT, name TEXT);
INSERT INTO cursor_case_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE Mixed_Case_Cursor CURSOR FOR SELECT id, name FROM cursor_case_people ORDER BY id;
FETCH FORWARD 1 FROM mixed_case_cursor;
MOVE FORWARD 1 FROM MIXED_CASE_CURSOR;
FETCH NEXT FROM Mixed_Case_Cursor;
CLOSE mixed_case_cursor;
FETCH NEXT FROM MIXED_CASE_CURSOR;
SELECT name FROM cursor_case_people WHERE id = 4;
