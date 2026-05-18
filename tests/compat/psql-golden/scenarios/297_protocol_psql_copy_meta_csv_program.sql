\echo === psql copy meta csv program recovery ===
CREATE TABLE copy_meta_csv_people (id INT, name TEXT);
\copy copy_meta_csv_people FROM PROGRAM 'printf "1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n"' WITH CSV
\copy copy_meta_csv_people TO PROGRAM 'cat' WITH CSV
SELECT name FROM copy_meta_csv_people WHERE id = 2;
