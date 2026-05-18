\echo === psql copy meta csv parenthesized delimiter program recovery ===
CREATE TABLE copy_meta_csv_delimiter_people (id INT, name TEXT);
\copy copy_meta_csv_delimiter_people FROM PROGRAM 'printf "id|name\n1|Ada\n2|\"Grace|Hopper\"\n"' WITH (FORMAT csv, HEADER, DELIMITER '|')
\copy copy_meta_csv_delimiter_people TO PROGRAM 'cat' WITH (FORMAT csv, HEADER, DELIMITER '|')
SELECT name FROM copy_meta_csv_delimiter_people WHERE id = 2;
