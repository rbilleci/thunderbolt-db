\echo === psql extended bind gset multirow recovery ===
CREATE TABLE ext_gset_multirow_people (id INT, name TEXT);
INSERT INTO ext_gset_multirow_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_gset_multirow_people WHERE id >= $1 ORDER BY id \bind 2 \gset multi_
SELECT name FROM ext_gset_multirow_people WHERE id = $1 \bind 1 \g
