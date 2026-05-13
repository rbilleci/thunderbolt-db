\echo === psql extended multirow bind select ===
CREATE TABLE ext_multi_people (id INT, name TEXT);
INSERT INTO ext_multi_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_multi_people WHERE id > $1 ORDER BY id \bind 1 \g
