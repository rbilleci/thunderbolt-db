\echo === psql gdesc copy unsupported recovery ===
CREATE TABLE gdesc_copy_people (id INT, name TEXT);
INSERT INTO gdesc_copy_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
/* leading comment before extended copy */ COPY gdesc_copy_people TO STDOUT \gdesc
SELECT name FROM gdesc_copy_people WHERE id = 2;
