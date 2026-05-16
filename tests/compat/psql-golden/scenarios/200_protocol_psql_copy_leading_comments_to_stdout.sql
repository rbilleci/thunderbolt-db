\echo === psql copy leading comments to stdout recovery ===
CREATE TABLE copy_comment_boundary_people (id INT, name TEXT);
INSERT INTO copy_comment_boundary_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
/* leading comment before supported copy */ COPY copy_comment_boundary_people TO STDOUT;
SELECT name FROM copy_comment_boundary_people WHERE id = 2;
