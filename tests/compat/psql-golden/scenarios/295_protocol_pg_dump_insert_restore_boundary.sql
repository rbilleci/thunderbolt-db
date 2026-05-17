\echo === pg_dump insert-style restore boundary ===
CREATE TABLE pg_dump_insert_people (id INT, name TEXT);
INSERT INTO public.pg_dump_insert_people VALUES (1, 'Ada'), (2, 'Grace');
SELECT id, name FROM ONLY public.pg_dump_insert_people ORDER BY id;
DECLARE pg_dump_insert_cursor CURSOR FOR SELECT id, name FROM ONLY public.pg_dump_insert_people;
FETCH ALL FROM pg_dump_insert_cursor;
CLOSE pg_dump_insert_cursor;
