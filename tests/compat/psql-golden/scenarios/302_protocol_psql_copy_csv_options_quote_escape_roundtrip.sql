\echo === psql copy csv parenthesized quote escape import export recovery ===
CREATE TABLE copy_csv_quote_escape_people (id INT, name TEXT);
COPY copy_csv_quote_escape_people FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER ';', QUOTE '|', ESCAPE '\');
id;name
1;Ada
2;|Grace;Hopper|
3;|Pipe \| and slash \\|
\.
SELECT id, name FROM copy_csv_quote_escape_people ORDER BY id;
COPY copy_csv_quote_escape_people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER ';', QUOTE '|', ESCAPE '\');
SELECT name FROM copy_csv_quote_escape_people WHERE id = 3;
