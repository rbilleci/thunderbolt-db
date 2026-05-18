\echo === drop constraints ===
CREATE TABLE dc_people (id INT PRIMARY KEY, name TEXT UNIQUE);
INSERT INTO dc_people (id, name) VALUES (1, 'Ada'), (2, 'Grace');
COMMENT ON INDEX public.dc_people_name_key IS 'name uniqueness';
COMMENT ON CONSTRAINT dc_people_pkey ON public.dc_people IS 'row identity';
\d dc_people
\dd dc_people_pkey
\dd dc_people_name_key
ALTER TABLE IF EXISTS ONLY public.dc_people DROP CONSTRAINT IF EXISTS dc_people_pkey;
ALTER TABLE ONLY public.dc_people DROP CONSTRAINT dc_people_name_key;
\d dc_people
\dd dc_people_pkey
\dd dc_people_name_key
INSERT INTO dc_people (id, name) VALUES (1, 'Ada');
SELECT id, name FROM dc_people ORDER BY id;
ALTER TABLE ONLY public.dc_people DROP CONSTRAINT IF EXISTS dc_people_pkey;
ALTER TABLE ONLY public.dc_people DROP CONSTRAINT dc_people_pkey;
ALTER TABLE IF EXISTS ONLY public.missing_dc_people DROP CONSTRAINT dc_people_pkey;
ALTER TABLE ONLY public.missing_dc_people DROP CONSTRAINT IF EXISTS dc_people_pkey;
ALTER TABLE ONLY public.dc_people DROP CONSTRAINT dc_people_pkey CASCADE;
SELECT id, name FROM dc_people ORDER BY id;
