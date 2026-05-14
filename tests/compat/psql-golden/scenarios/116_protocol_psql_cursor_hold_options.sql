\echo === psql cursor hold options ===
CREATE TABLE cursor_hold_people (id INT, name TEXT);
INSERT INTO cursor_hold_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE without_hold_people CURSOR WITHOUT HOLD FOR SELECT id, name FROM cursor_hold_people ORDER BY id;
FETCH FORWARD 2 FROM without_hold_people;
CLOSE without_hold_people;
DECLARE no_scroll_without_hold_people NO SCROLL CURSOR WITHOUT HOLD FOR SELECT id, name FROM cursor_hold_people ORDER BY id;
FETCH FORWARD 1 FROM no_scroll_without_hold_people;
CLOSE no_scroll_without_hold_people;
DECLARE hold_people CURSOR WITH HOLD FOR SELECT id, name FROM cursor_hold_people ORDER BY id;
FETCH FORWARD 1 FROM hold_people;
SELECT name FROM cursor_hold_people WHERE id = 2;
