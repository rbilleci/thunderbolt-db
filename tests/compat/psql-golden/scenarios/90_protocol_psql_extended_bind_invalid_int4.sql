\echo === psql extended bind invalid int4 recovery ===
CREATE TABLE ext_bind_invalid_people (id INT, name TEXT);
INSERT INTO ext_bind_invalid_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM ext_bind_invalid_people WHERE id = $1 \bind not-an-int \g
SELECT name FROM ext_bind_invalid_people WHERE id = $1 \bind 2 \g
