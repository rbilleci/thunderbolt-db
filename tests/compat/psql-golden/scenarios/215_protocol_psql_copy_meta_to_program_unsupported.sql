\echo === psql copy meta to program unsupported recovery ===
CREATE TABLE copy_meta_to_people (id INT, name TEXT);
INSERT INTO copy_meta_to_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
\copy copy_meta_to_people TO PROGRAM 'cat >/dev/null'
SELECT name FROM copy_meta_to_people WHERE id = 1;
