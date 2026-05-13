\echo === psql extended reused placeholder bind ===
CREATE TABLE ext_reused_placeholder_people (id INT, name TEXT);
INSERT INTO ext_reused_placeholder_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
SELECT id, name FROM ext_reused_placeholder_people WHERE id >= $1 AND id <= $1 ORDER BY id \bind not-an-int \g
SELECT id, name FROM ext_reused_placeholder_people WHERE id >= $1 AND id <= $1 ORDER BY id \bind 2 \g
SELECT id, name FROM ext_reused_placeholder_people WHERE id >= 2 ORDER BY id;
