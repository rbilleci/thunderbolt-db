\echo === psql extended negative limit bind parameter ===
CREATE TABLE ext_negative_limit_people (id INT, name TEXT);
INSERT INTO ext_negative_limit_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_negative_limit_people ORDER BY id LIMIT $1 \bind -1 \g
SELECT id, name FROM ext_negative_limit_people ORDER BY id LIMIT $1 \bind 0 \g
SELECT id, name FROM ext_negative_limit_people ORDER BY id LIMIT $1 \bind 2 \g
SELECT name FROM ext_negative_limit_people WHERE id = 3;
