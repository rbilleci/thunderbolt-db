\echo === psql direct cursor negative limit recovery ===
CREATE TABLE cursor_direct_negative_people (id INT, name TEXT);
INSERT INTO cursor_direct_negative_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE cursor_direct_negative_bad CURSOR FOR SELECT id, name FROM cursor_direct_negative_people ORDER BY id LIMIT -1;
DECLARE cursor_direct_negative_ok CURSOR FOR SELECT id, name FROM cursor_direct_negative_people ORDER BY id LIMIT 2;
FETCH ALL FROM cursor_direct_negative_ok;
CLOSE cursor_direct_negative_ok;
SELECT name FROM cursor_direct_negative_people WHERE id = 3;
