\echo === psql copy to stdout export recovery ===
CREATE TABLE copy_boundary_people (id INT, name TEXT);
INSERT INTO copy_boundary_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COPY copy_boundary_people TO STDOUT;
SELECT name FROM copy_boundary_people WHERE id = 2;
