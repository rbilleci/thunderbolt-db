\echo === psql extended parameterized gdesc ===
CREATE TABLE ext_gdesc_people (id INT, name TEXT);
INSERT INTO ext_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM ext_gdesc_people WHERE id = $1 \gdesc
SELECT name FROM ext_gdesc_people WHERE id = 2;
