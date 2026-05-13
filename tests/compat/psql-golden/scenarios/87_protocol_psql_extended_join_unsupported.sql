\echo === psql extended join unsupported ===
CREATE TABLE ext_join_people (id INT, name TEXT);
CREATE TABLE ext_join_pets (owner_id INT, pet TEXT);
INSERT INTO ext_join_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
INSERT INTO ext_join_pets (owner_id, pet) VALUES (1, 'Byte'), (2, 'Tux');
SELECT id FROM ext_join_people JOIN ext_join_pets ON id = owner_id WHERE id = $1 \bind 1 \g
SELECT id, name FROM ext_join_people WHERE id = 2;
