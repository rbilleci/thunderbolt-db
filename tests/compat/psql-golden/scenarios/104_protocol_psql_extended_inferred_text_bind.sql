\echo === psql extended inferred text bind parameter type ===
CREATE TABLE ext_infer_text_people (id INT, name TEXT);
INSERT INTO ext_infer_text_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Ada');
SELECT id, name FROM ext_infer_text_people WHERE name = $1 ORDER BY id \bind Ada \g
SELECT id, name FROM ext_infer_text_people WHERE name = $1 ORDER BY id \bind Missing \g
