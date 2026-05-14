\echo === psql cursor prior and backward variants ===
CREATE TABLE cursor_prior_people (id INT, name TEXT);
INSERT INTO cursor_prior_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
DECLARE cursor_prior_people CURSOR FOR SELECT id, name FROM cursor_prior_people ORDER BY id;
FETCH FORWARD 2 FROM cursor_prior_people;
FETCH PRIOR FROM cursor_prior_people;
FETCH NEXT FROM cursor_prior_people;
FETCH BACKWARD ALL FROM cursor_prior_people;
MOVE PRIOR FROM cursor_prior_people;
MOVE BACKWARD ALL FROM cursor_prior_people;
FETCH NEXT FROM cursor_prior_people;
SELECT name FROM cursor_prior_people WHERE id = 4;
