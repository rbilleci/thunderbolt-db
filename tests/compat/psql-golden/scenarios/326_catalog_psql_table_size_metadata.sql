\echo === catalog psql table size metadata ===
CREATE TABLE heap_size_people (id INT, name TEXT);
\dt+ heap_size_people
INSERT INTO heap_size_people (id, name) VALUES (1, 'ada'), (2, 'grace');
\dt+ heap_size_people
\dt+ heap_size_*
\dt+ public.heap_size_people
\dt+ public.heap_size_*
