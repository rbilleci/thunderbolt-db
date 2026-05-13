\echo === psql extended bind predicate and limit parameters ===
CREATE TABLE ext_limit_people (id INT, name TEXT);
INSERT INTO ext_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Edsger');
SELECT id, name FROM ext_limit_people WHERE id >= $1 ORDER BY id LIMIT $2 \bind 2 2 \g
SELECT id, name FROM ext_limit_people WHERE id >= $1 ORDER BY id LIMIT $2 \bind 3 10 \g
