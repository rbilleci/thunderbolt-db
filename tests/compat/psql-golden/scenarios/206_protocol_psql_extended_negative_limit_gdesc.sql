\echo === psql extended negative limit gdesc ===
CREATE TABLE ext_negative_limit_gdesc_people (id INT, name TEXT);
INSERT INTO ext_negative_limit_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_negative_limit_gdesc_people WHERE id >= $1 ORDER BY id LIMIT $2 \bind 1 -1 \gdesc
SELECT id, name FROM ext_negative_limit_gdesc_people WHERE id >= $1 ORDER BY id LIMIT $2 \bind 1 -1 \g
SELECT id, name FROM ext_negative_limit_gdesc_people WHERE id >= $1 ORDER BY id LIMIT $2 \bind 1 2 \g
SELECT name FROM ext_negative_limit_gdesc_people WHERE id = 3;
