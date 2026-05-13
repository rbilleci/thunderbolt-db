\echo === psql extended bind ignores commented placeholders ===
CREATE TABLE ext_comment_placeholder_people (id INT, name TEXT);
INSERT INTO ext_comment_placeholder_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, '$2');
SELECT id, name FROM ext_comment_placeholder_people WHERE id >= $1 -- ignored $2
ORDER BY id LIMIT $2 \bind 2 1 \g
SELECT id, name FROM ext_comment_placeholder_people WHERE name = '/* $1 */' OR id = $1 /* ignored $2 /* nested ignored $3 */ still ignored $4 */ \bind 1 \g
