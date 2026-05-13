\echo === psql extended text bind and empty result ===
CREATE TABLE ext_text_bind_people (id INT, name TEXT);
INSERT INTO ext_text_bind_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id FROM ext_text_bind_people WHERE name = $1 \bind Linus \g
SELECT id FROM ext_text_bind_people WHERE name = $1 \bind Missing \g
