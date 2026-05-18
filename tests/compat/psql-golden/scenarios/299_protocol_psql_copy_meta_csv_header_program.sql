\echo === psql copy meta csv header program recovery ===
CREATE TABLE copy_meta_csv_header_people (id INT, name TEXT);
\copy copy_meta_csv_header_people FROM PROGRAM 'printf "id,name\n1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n"' WITH CSV HEADER
\copy copy_meta_csv_header_people TO PROGRAM 'cat' WITH CSV HEADER
SELECT name FROM copy_meta_csv_header_people WHERE id = 2;
