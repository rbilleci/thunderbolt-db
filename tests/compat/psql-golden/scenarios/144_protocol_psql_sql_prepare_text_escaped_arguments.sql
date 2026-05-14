\echo === psql SQL prepare escaped text arguments ===
CREATE TABLE sql_prepare_text_people (id INT, name TEXT);
INSERT INTO sql_prepare_text_people (id, name) VALUES (1, 'Ada'), (2, 'O''Brien, Sr.'), (3, 'Grace');
PREPARE sql_prepare_text_lookup(text) AS SELECT id, name FROM sql_prepare_text_people WHERE name = $1;
EXECUTE sql_prepare_text_lookup('O''Brien, Sr.');
EXECUTE sql_prepare_text_lookup('Missing, Sr.');
DEALLOCATE sql_prepare_text_lookup;
EXECUTE sql_prepare_text_lookup('O''Brien, Sr.');
SELECT name FROM sql_prepare_text_people WHERE id = 3;
