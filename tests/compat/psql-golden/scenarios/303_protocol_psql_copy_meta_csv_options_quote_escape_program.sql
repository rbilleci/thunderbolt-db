\echo === psql copy meta csv parenthesized quote escape program recovery ===
CREATE TABLE copy_meta_csv_quote_escape_people (id INT, name TEXT);
\copy copy_meta_csv_quote_escape_people FROM PROGRAM 'printf "%s\n" "id;name" "1;Ada" "2;|Grace;Hopper|" "3;|Pipe \\| and slash \\\\|"' WITH (FORMAT csv, HEADER, DELIMITER ';', QUOTE '|', ESCAPE '\')
\copy copy_meta_csv_quote_escape_people TO PROGRAM 'cat' WITH (FORMAT csv, HEADER, DELIMITER ';', QUOTE '|', ESCAPE '\')
SELECT name FROM copy_meta_csv_quote_escape_people WHERE id = 3;
