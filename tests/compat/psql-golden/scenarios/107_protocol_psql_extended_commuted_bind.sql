\echo === psql extended commuted placeholder bind ===
CREATE TABLE ext_commuted_people (id INT, name TEXT);
INSERT INTO ext_commuted_people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Linus');
SELECT id, name FROM ext_commuted_people WHERE $1 <= id AND $2 = name ORDER BY id \bind not-an-int Grace \g
SELECT id, name FROM ext_commuted_people WHERE $1 <= id AND $2 = name ORDER BY id \bind 2 Grace \g
SELECT id, name FROM ext_commuted_people WHERE id >= 2 ORDER BY id;
