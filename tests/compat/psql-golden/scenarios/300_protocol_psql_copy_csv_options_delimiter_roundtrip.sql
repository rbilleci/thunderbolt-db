\echo === psql copy csv parenthesized delimiter import export recovery ===
CREATE TABLE copy_csv_delimiter_people (id INT, name TEXT);
COPY copy_csv_delimiter_people FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|');
id|name
1|Ada
2|"Grace|Hopper"
3|Comma, Literal
\.
SELECT id, name FROM copy_csv_delimiter_people ORDER BY id;
COPY copy_csv_delimiter_people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|');
SELECT name FROM copy_csv_delimiter_people WHERE id = 2;
