\echo === psql extended inferred OR bind parameter types ===
CREATE TABLE ext_or_people (id INT, name TEXT);
INSERT INTO ext_or_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus'), (4, 'Ada');
SELECT id, name FROM ext_or_people WHERE id = $1 OR name = $2 ORDER BY id \bind not-an-int Grace \g
SELECT id, name FROM ext_or_people WHERE id = $1 OR name = $2 ORDER BY id \bind 1 Grace \g
SELECT id, name FROM ext_or_people WHERE id >= 3 ORDER BY id;
