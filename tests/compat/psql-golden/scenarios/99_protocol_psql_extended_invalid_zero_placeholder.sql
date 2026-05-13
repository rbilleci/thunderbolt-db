\echo === psql extended zero placeholder rejection ===
CREATE TABLE ext_zero_placeholder_people (id INT, name TEXT);
INSERT INTO ext_zero_placeholder_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM ext_zero_placeholder_people WHERE id = $0 \bind \g
SELECT name FROM ext_zero_placeholder_people WHERE id = 2;
