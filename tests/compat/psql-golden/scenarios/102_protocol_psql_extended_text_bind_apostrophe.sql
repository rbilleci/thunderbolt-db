\echo === psql extended text bind apostrophe escaping ===
CREATE TABLE ext_text_apostrophe_people (id INT, name TEXT);
INSERT INTO ext_text_apostrophe_people (id, name) VALUES (1, 'Ada'), (2, 'O''Brien'), (3, 'Grace');
SELECT id FROM ext_text_apostrophe_people WHERE name = $1 \bind 'O\'Brien' \g
SELECT id FROM ext_text_apostrophe_people WHERE name = $1 \bind 'Missing\'OBrien' \g
SELECT name FROM ext_text_apostrophe_people WHERE id = 3;
