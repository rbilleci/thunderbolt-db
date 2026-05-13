\echo === psql extended bind parameter count mismatch ===
CREATE TABLE ext_bind_count_people (id INT, name TEXT);
INSERT INTO ext_bind_count_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM ext_bind_count_people WHERE id = $1 AND name = $2 \bind 1 \g
SELECT name FROM ext_bind_count_people WHERE id = $1 AND name = $2 \bind 2 Linus \g
