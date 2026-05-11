\echo === extended protocol bind execute ===
CREATE TABLE ext_people (id INT, name TEXT);
INSERT INTO ext_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name, id FROM ext_people WHERE id = $1 ORDER BY name DESC LIMIT 1 \bind 2 \g
\echo === extended named prepared lifecycle ===
SELECT name FROM ext_people WHERE id = $1 \parse ext_lookup
\bind_named ext_lookup 3 \g
\close_prepared ext_lookup
