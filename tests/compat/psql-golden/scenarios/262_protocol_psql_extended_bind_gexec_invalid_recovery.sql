\echo === psql extended bind gexec invalid recovery ===
CREATE TABLE ext_bind_gexec_invalid_people (id INT, name TEXT);
CREATE TABLE ext_bind_gexec_invalid_commands (id INT, command TEXT);
INSERT INTO ext_bind_gexec_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
INSERT INTO ext_bind_gexec_invalid_commands (id, command) VALUES (1, 'SELECT name FROM ext_bind_gexec_invalid_people WHERE id = 2;');
SELECT command FROM ext_bind_gexec_invalid_commands WHERE id = $1 \bind not_an_int \gexec
SELECT command FROM ext_bind_gexec_invalid_commands WHERE id = $1 \bind 1 \gexec
SELECT name FROM ext_bind_gexec_invalid_people WHERE id = 3;
