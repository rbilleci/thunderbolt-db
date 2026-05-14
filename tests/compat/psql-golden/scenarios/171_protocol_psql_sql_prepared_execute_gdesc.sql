\echo === psql sql prepared execute gdesc ===
CREATE TABLE exec_gdesc_people (id INT, name TEXT);
INSERT INTO exec_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE exec_gdesc_lookup(int4) AS SELECT name FROM exec_gdesc_people WHERE id = $1;
EXECUTE exec_gdesc_lookup(2) \gdesc
EXECUTE exec_gdesc_lookup(2);
SELECT name FROM exec_gdesc_people WHERE id = 1;
