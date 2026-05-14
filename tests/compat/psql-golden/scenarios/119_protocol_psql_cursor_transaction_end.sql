\echo === psql cursor transaction end lifecycle ===
CREATE TABLE cursor_tx_people (id INT, name TEXT);
INSERT INTO cursor_tx_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
BEGIN;
DECLARE commit_people NO SCROLL CURSOR WITHOUT HOLD FOR SELECT id, name FROM cursor_tx_people ORDER BY id;
FETCH FORWARD 1 FROM commit_people;
COMMIT;
FETCH FORWARD 1 FROM commit_people;
BEGIN;
DECLARE rollback_people CURSOR FOR SELECT id, name FROM cursor_tx_people ORDER BY id;
FETCH FORWARD 2 FROM rollback_people;
ROLLBACK;
FETCH FORWARD 1 FROM rollback_people;
SELECT name FROM cursor_tx_people WHERE id = 3;
