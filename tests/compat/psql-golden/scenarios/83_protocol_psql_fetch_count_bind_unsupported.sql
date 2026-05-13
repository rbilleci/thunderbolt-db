\echo === psql fetch count bind cursor boundary ===
\set FETCH_COUNT 2
CREATE TABLE ext_fetch_param_people (id INT, name TEXT);
INSERT INTO ext_fetch_param_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_fetch_param_people WHERE id > $1 ORDER BY id \bind 0 \g
\set FETCH_COUNT 0
SELECT name FROM ext_fetch_param_people WHERE id = 2;
