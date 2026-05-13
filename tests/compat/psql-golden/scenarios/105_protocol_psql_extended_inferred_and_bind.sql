\echo === psql extended inferred AND bind parameter types ===
CREATE TABLE ext_infer_and_people (id INT, name TEXT);
INSERT INTO ext_infer_and_people (id, name) VALUES (1, 'Ada'), (2, 'Ada'), (3, 'Linus'), (4, 'Ada');
SELECT id, name FROM ext_infer_and_people WHERE id >= $1 AND name = $2 ORDER BY id \bind not-an-int Ada \g
SELECT id, name FROM ext_infer_and_people WHERE id >= $1 AND name = $2 ORDER BY id \bind 2 Ada \g
SELECT id, name FROM ext_infer_and_people WHERE id >= 3 ORDER BY id;
