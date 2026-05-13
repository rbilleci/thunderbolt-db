\echo === psql extended inferred bind parameter types ===
CREATE TABLE ext_infer_people (id INT, name TEXT);
INSERT INTO ext_infer_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT name FROM ext_infer_people WHERE id = $1 \bind not-an-int \g
SELECT name FROM ext_infer_people WHERE id = $1 \bind 2 \g
