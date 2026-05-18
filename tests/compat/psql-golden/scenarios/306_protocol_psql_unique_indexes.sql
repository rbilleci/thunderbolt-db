\echo === unique index enforcement ===
CREATE TABLE unique_people (id INT, name TEXT);
INSERT INTO unique_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
CREATE UNIQUE INDEX unique_people_name_uidx ON unique_people (name);
SELECT schemaname, tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;
\di+
INSERT INTO unique_people (id, name) VALUES (3, 'Ada');
SELECT id, name FROM unique_people ORDER BY id;
UPDATE unique_people SET name = 'Ada' WHERE id = 2;
SELECT id, name FROM unique_people ORDER BY id;
COPY unique_people FROM STDIN WITH CSV;
4,Ada
\.
SELECT id, name FROM unique_people ORDER BY id;
CREATE TABLE duplicate_unique_people (id INT, name TEXT);
INSERT INTO duplicate_unique_people (id, name) VALUES (1, 'Ada'), (2, 'Ada');
CREATE UNIQUE INDEX duplicate_unique_people_name_uidx ON duplicate_unique_people (name);
