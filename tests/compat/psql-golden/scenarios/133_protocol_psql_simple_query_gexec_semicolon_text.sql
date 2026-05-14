\echo === psql gexec semicolon text ===
CREATE TABLE gexec_semicolon_people (id INT, name TEXT);
CREATE TABLE gexec_semicolon_commands (id INT, command TEXT);
INSERT INTO gexec_semicolon_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO gexec_semicolon_commands (id, command) VALUES (1, 'SELECT name FROM gexec_semicolon_people WHERE id = 2;'), (2, 'SELECT name FROM gexec_semicolon_people WHERE id = 3;');
SELECT command FROM gexec_semicolon_commands WHERE id = 1 \gexec
SELECT name FROM gexec_semicolon_people WHERE id = 1;
