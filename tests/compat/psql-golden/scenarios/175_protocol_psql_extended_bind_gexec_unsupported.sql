\echo === psql extended bind gexec ===
CREATE TABLE ext_gexec_people (id INT, name TEXT);
CREATE TABLE ext_gexec_commands (id INT, command TEXT);
INSERT INTO ext_gexec_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO ext_gexec_commands (id, command) VALUES (1, 'SELECT name FROM ext_gexec_people WHERE id = 2;'), (2, 'SELECT name FROM ext_gexec_people WHERE id = 3;');
SELECT command FROM ext_gexec_commands WHERE id = $1 \bind 1 \gexec
SELECT name FROM ext_gexec_people WHERE id = 1;
