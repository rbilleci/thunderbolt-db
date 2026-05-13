\echo === psql extended bind multi-digit placeholders ===
CREATE TABLE ext_multidigit_placeholder_people (id INT, name TEXT);
INSERT INTO ext_multidigit_placeholder_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (10, 'Grace'), (11, 'Edsger');
SELECT id, name FROM ext_multidigit_placeholder_people WHERE id = $1 OR id = $10 ORDER BY id LIMIT $11 \bind 1 2 3 4 5 6 7 8 9 10 10 \g
