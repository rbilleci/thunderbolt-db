\echo === psql extended gdesc bind error recovery ===
CREATE TABLE ext_gdesc_recovery_people (id INT, name TEXT);
INSERT INTO ext_gdesc_recovery_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
SELECT name FROM ext_gdesc_recovery_people WHERE id = $1 \bind nope \g
SELECT name FROM ext_gdesc_recovery_people WHERE id = $1 \bind 2 \gdesc
SELECT name FROM ext_gdesc_recovery_people WHERE id = $1 \bind 2 \g
SELECT name FROM ext_gdesc_recovery_people WHERE id = 1;
