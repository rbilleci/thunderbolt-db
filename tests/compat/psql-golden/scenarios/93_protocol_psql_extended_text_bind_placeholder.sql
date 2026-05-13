\echo === psql extended text bind placeholder literal ===
CREATE TABLE ext_text_placeholder_people (id INT, name TEXT);
INSERT INTO ext_text_placeholder_people (id, name) VALUES (1, 'Ada$1'), (2, 'Linus'), (3, 'Grace');
SELECT id FROM ext_text_placeholder_people WHERE name = $1 \bind Ada$1 \g
SELECT id FROM ext_text_placeholder_people WHERE name = $1 \bind Missing$1 \g
