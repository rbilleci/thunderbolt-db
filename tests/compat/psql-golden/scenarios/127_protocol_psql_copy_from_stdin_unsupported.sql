\echo === psql copy from stdin unsupported recovery ===
CREATE TABLE copy_from_boundary_people (id INT, name TEXT);
INSERT INTO copy_from_boundary_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COPY copy_from_boundary_people FROM STDIN;
SELECT name FROM copy_from_boundary_people WHERE id = 1;
