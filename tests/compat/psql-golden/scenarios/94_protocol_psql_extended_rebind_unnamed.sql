\echo === psql extended unnamed rebind execution ===
CREATE TABLE ext_rebind_people (id INT, name TEXT);
INSERT INTO ext_rebind_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM ext_rebind_people WHERE id = $1 \bind 1 \g
SELECT name FROM ext_rebind_people WHERE id = $1 \bind 3 \g
SELECT name FROM ext_rebind_people WHERE id = $1 \bind 2 \gdesc
SELECT name FROM ext_rebind_people WHERE id = $1 \bind 2 \g
