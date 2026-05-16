\echo === psql copy from stdin import recovery ===
CREATE TABLE copy_from_boundary_people (id INT, name TEXT);
COPY copy_from_boundary_people FROM STDIN;
1	Ada
2	Linus
3	Grace\tHopper
\.
SELECT name FROM copy_from_boundary_people WHERE id = 3;
