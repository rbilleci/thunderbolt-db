\echo === psql extended bind quoted placeholder literals ===
CREATE TABLE ext_quoted_placeholder_people (id INT, name TEXT);
INSERT INTO ext_quoted_placeholder_people (id, name) VALUES (1, '$1'), (2, 'Linus'), (3, '$2');
SELECT id, name FROM ext_quoted_placeholder_people WHERE name = '$1' OR id = $1 ORDER BY id \bind 2 \g
SELECT id, name FROM ext_quoted_placeholder_people WHERE name = '$2' OR name = $1 ORDER BY id \bind Linus \g
