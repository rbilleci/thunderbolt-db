\echo === psql gdesc copy from unsupported recovery ===
CREATE TABLE gdesc_copy_from_people (id INT, name TEXT);
INSERT INTO gdesc_copy_from_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
COPY gdesc_copy_from_people FROM STDIN \gdesc
SELECT name FROM gdesc_copy_from_people WHERE id = 1;
