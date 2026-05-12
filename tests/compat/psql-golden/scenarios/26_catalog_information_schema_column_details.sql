\echo === information_schema per-table column details ===
CREATE TABLE is_column_details_people (id INT, name TEXT);
SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'is_column_details_people' ORDER BY ordinal_position;
