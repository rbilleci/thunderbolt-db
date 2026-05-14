\echo === psql copy meta from stdin unsupported recovery ===
CREATE TABLE copy_meta_people (id INT, name TEXT);
INSERT INTO copy_meta_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
\copy copy_meta_people FROM PROGRAM 'printf "3\tGrace\n"'
SELECT name FROM copy_meta_people WHERE id = 2;
