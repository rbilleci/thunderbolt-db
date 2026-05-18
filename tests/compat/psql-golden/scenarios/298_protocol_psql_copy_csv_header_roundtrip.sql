\echo === psql copy csv header import export recovery ===
CREATE TABLE copy_csv_header_people (id INT, name TEXT);
COPY copy_csv_header_people FROM STDIN WITH CSV HEADER;
id,name
1,Ada
2,"Grace, ""Hopper"""
3,"Line\tLiteral"
\.
SELECT id, name FROM copy_csv_header_people ORDER BY id;
COPY copy_csv_header_people TO STDOUT WITH CSV HEADER;
SELECT name FROM copy_csv_header_people WHERE id = 2;
