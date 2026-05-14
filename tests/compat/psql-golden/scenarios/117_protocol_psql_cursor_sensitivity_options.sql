\echo === psql cursor sensitivity options ===
CREATE TABLE cursor_sensitivity_people (id INT, name TEXT);
INSERT INTO cursor_sensitivity_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
DECLARE asensitive_people ASENSITIVE NO SCROLL CURSOR FOR SELECT id, name FROM cursor_sensitivity_people ORDER BY id;
FETCH FORWARD 2 FROM asensitive_people;
CLOSE asensitive_people;
DECLARE insensitive_people INSENSITIVE CURSOR WITHOUT HOLD FOR SELECT id, name FROM cursor_sensitivity_people ORDER BY id;
FETCH ALL FROM insensitive_people;
CLOSE insensitive_people;
DECLARE binary_insensitive_people BINARY INSENSITIVE CURSOR FOR SELECT id, name FROM cursor_sensitivity_people ORDER BY id;
FETCH FORWARD 1 FROM binary_insensitive_people;
SELECT name FROM cursor_sensitivity_people WHERE id = 3;
